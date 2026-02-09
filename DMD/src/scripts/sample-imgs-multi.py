import os
import pathlib
import sys

script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunks_num(lst, n):
    """Yield n evenly sized chunks from lst."""
    low = len(lst)//n
    rem = len(lst)-(low*n)
    counts = [low]*n
    for i in range(rem): counts[i] += 1
    ptr = 0
    res = []
    for count in counts:
        res.append(lst[ptr:ptr+count])
        ptr += count
    return res

def gpu_list(string):
    return [int(item) for item in string.split(',')]

import argparse
argParser = argparse.ArgumentParser(description="start/resume inference on a given json file")
argParser.add_argument("config")
argParser.add_argument("-g","--gpus",dest="gpus",action="store",default=[0],type=gpu_list, help="Comma-separated list of GPU IDs to use")
argParser.add_argument("-s",dest="steps",action="store",default=500,type=int,help="Number of steps of diffusion to run for each sample")
argParser.add_argument("-b","--batch_size",dest="batch_size",action="store",default=16,type=int,help="Batch size for sampling")
argParser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
argParser.add_argument("--model", help="Path to model checkpoint",required=True)
argParser.add_argument('--sfm_method', type=str)
argParser.add_argument("--left_right_pose_cond", action="store_true", help="Whether to use pose conditioning for aTc and cTa in CrossAttnBlock")
argParser.add_argument("--gripper_state", action="store_true", help="Whether to include gripper state in the input")
argParser.add_argument("--stable_diffusion_encoder", action="store_true", help="Whether to use stable diffusion encoder rather than the VQGAN")
argParser.add_argument("--external_gpu", action="store_true", help="Whether to use the external GPU to train")
argParser.add_argument('--image_size', type=int, default=128, help="image size")
cli_args = argParser.parse_args()
global_sfm_method = cli_args.sfm_method
assert cli_args.model is not None

from utils import *

from models.score_sde import ncsnpp
import models.score_sde.sde_lib as sde_lib
from models.score_sde.ncsnpp_dual import NCSNpp_dual, NCSNpp_dual_bimanual
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
import json
from models.score_sde.configs.LDM import LDM_config
import shutil
from models.vqgan.vqgan import VQModel as VQGAN
from models.vqgan.configs.vqgan_32_4 import vqgan_32_4_config
from models.latentdiffusion.sb_32_4 import sb_32_4_config
from models.score_sde.ema import ExponentialMovingAverage
import functools
from models.score_sde.layerspp import ResnetBlockBigGANpp
import torch.nn.functional as F
import pudb
from filelock import FileLock
import time
from cv2 import cv2
# center crop image to shape
def center_crop(img,shape):
    h,w = shape
    center = img.shape
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    center_crop = img[int(y):int(y+h), int(x):int(x+w)]
    return center_crop

def maximal_crop_to_shape(image,shape,interpolation=cv2.INTER_AREA):
    target_aspect = shape[1]/shape[0]
    input_aspect = image.shape[1]/image.shape[0]
    if input_aspect > target_aspect:
        center_crop_shape = (image.shape[0],int(image.shape[0] * target_aspect))
    else:
        center_crop_shape = (int(image.shape[1] / target_aspect),image.shape[1])
    cropped = center_crop(image,center_crop_shape)
    resized = cv2.resize(cropped, (shape[1],shape[0]),interpolation=interpolation)
    return resized

def crop_like_dataset(frame_a):
	orig_w, orig_h = frame_a.size
	if orig_w == 256 and orig_h == 256:
		return np.asarray(frame_a)
	# ensure images have 360 height
	if orig_h != 360:
		new_w = round(frame_a.size[0] * (360/frame_a.size[1]))
		frame_a = np.asarray(frame_a.resize((new_w,360)))
	else:
		frame_a = np.asarray(frame_a)
	if orig_w != 360: # if the width is not 360
		# crop and downsample
		left_pos = (frame_a.shape[1]-360)//2
		frame_a_cropped = frame_a[:,left_pos:left_pos+360,:]
	else:
		frame_a_cropped = frame_a
	im_a = Image.fromarray(frame_a_cropped).resize((256,256))
	return np.asarray(im_a)

def inference(data,device):
	# vqgan
	vqgan = None
	if not cli_args.stable_diffusion_encoder:
		vqgan = VQGAN(**vqgan_32_4_config).to(device)
	else:
		from ldm.models.autoencoder import VQModel as VQ_LDM
		vqgan = VQ_LDM(**sb_32_4_config).to(device)
	vqgan.eval()

	# ========================= build model =========================
	config = LDM_config()
	if global_sfm_method == 'orbslam_bimanual':
		if cli_args.left_right_pose_cond:
			config.left_right_pose_cond = cli_args.left_right_pose_cond
		config.gripper_state = cli_args.gripper_state
		score_model = NCSNpp_dual_bimanual(config)
		score_model.to(device)
		sde_bimanual = sde_lib.VESDE_Bimanual(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
	else:
		score_model = NCSNpp_dual(config)
		score_model.to(device)
		sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=cli_args.steps)

	# ray downsampler
	ResnetBlock = functools.partial(ResnetBlockBigGANpp,
									act=torch.nn.SiLU(),
									dropout=False,
									fir=True,
									fir_kernel=[1,3,3,1],
									init_scale=0,
									skip_rescale=True,
									temb_dim=None)
	ray_downsampler = torch.nn.Sequential(
		ResnetBlock(in_ch=56,out_ch=128,down=True),
		ResnetBlock(in_ch=128,out_ch=128,down=True)).to(device)
	if global_sfm_method == 'orbslam_bimanual':
		score_sde = Score_sde_bimanual_model(score_model, sde_bimanual, ray_downsampler, rays_require_downsample=False, rays_as_list=True)
	else:
		score_sde = Score_sde_model(score_model,sde,ray_downsampler,rays_require_downsample=False,rays_as_list=True)

	checkpoint = torch.load(cli_args.model)
	adapted_state = {}
	for k,v in checkpoint['score_sde_model'].items():
		key_parts = k.split('.')
		if key_parts[1] == 'module':
			key_parts.pop(1)
		new_key  = '.'.join(key_parts)
		adapted_state[new_key] = v
	score_sde.load_state_dict(adapted_state)

	ema = ExponentialMovingAverage(score_sde.parameters(),decay=0.999)
	ema.load_state_dict(checkpoint['ema'])
	ema.copy_to(score_sde.parameters())
	# substitute model with score modifier, used to make bulk sampling easier
	if global_sfm_method == 'orbslam_bimanual':
		modifier = Score_modifier_bimanual(score_sde.score_model,max_batch_size=cli_args.batch_size)
	else:
		modifier = Score_modifier(score_sde.score_model,max_batch_size=cli_args.batch_size)
	score_sde.score_model = modifier

	tlog('Setup complete','note')
	batches = chunks(data, cli_args.batch_size)

	if global_sfm_method == 'orbslam_bimanual':
		with torch.no_grad():
			for batch in batches:
				if cli_args.verbose:
					for el in batch:
						print(device, el['img'])
				w_l_conditioning_ims = []
				w_l_ff_refs = []
				w_l_ff_as = []
				w_l_ff_bs = []
				w_r_conditioning_ims = []
				w_r_ff_refs = []
				w_r_ff_as = []
				w_r_ff_bs = []
				gripper_state = []
				for el in batch:
					left_im = Image.open(el['left_img'])
					left_im = crop_like_dataset(left_im)
					if not os.path.exists(el['left_orig_img_copy_path']):
						cv2.imwrite(el['left_orig_img_copy_path'],cv2.cvtColor(left_im, cv2.COLOR_RGB2BGR))
					left_im = left_im[:,:,:3].astype(np.float32).transpose(2,0,1)/127.5 - 1
					left_im = torch.Tensor(left_im).unsqueeze(0).to(device)
					left_encoded_im = None
					if not cli_args.stable_diffusion_encoder:
						left_encoded_im = vqgan.encode(left_im)
					else:
						left_encoded_im = vqgan.encode(left_im)[0]

					right_im = Image.open(el['right_img'])
					right_im = crop_like_dataset(right_im)
					if not os.path.exists(el['right_orig_img_copy_path']):
						cv2.imwrite(el['right_orig_img_copy_path'],cv2.cvtColor(right_im, cv2.COLOR_RGB2BGR))
					right_im = right_im[:,:,:3].astype(np.float32).transpose(2,0,1)/127.5 - 1
					right_im = torch.Tensor(right_im).unsqueeze(0).to(device)
					right_encoded_im = None
					if not cli_args.stable_diffusion_encoder:
						right_encoded_im = vqgan.encode(right_im)
					else:
						right_encoded_im = vqgan.encode(right_im)[0]

					# left wrist
					if 'wl_focal_y' in el:
						wl_focal_y = el['wl_focal_y']
					else:
						wl_focal_y = el['focal_y']
					w_l_tform_ref = np.eye(4)
					w_l_tform_a_relative = np.asarray(el['transformation_left'])
					w_l_tform_b_relative = np.linalg.inv(np.asarray(el['transformation_left']))
					w_l_camera_enc_ref = rel_camera_ray_encoding(w_l_tform_ref,cli_args.image_size,wl_focal_y)
					w_l_camera_enc_a = rel_camera_ray_encoding(w_l_tform_a_relative,cli_args.image_size,wl_focal_y)
					w_l_camera_enc_b = rel_camera_ray_encoding(w_l_tform_b_relative,cli_args.image_size,wl_focal_y)
					w_l_camera_enc_ref = torch.Tensor(w_l_camera_enc_ref).unsqueeze(0)
					w_l_camera_enc_a = torch.Tensor(w_l_camera_enc_a).unsqueeze(0)
					w_l_camera_enc_b = torch.Tensor(w_l_camera_enc_b).unsqueeze(0)
					w_l_ff_ref = F.pad(freq_enc(w_l_camera_enc_ref),[0,0,0,0,1,1,0,0]) # pad, must be %4==0 for group norm
					w_l_ff_a = F.pad(freq_enc(w_l_camera_enc_a),[0,0,0,0,1,1,0,0])
					w_l_ff_b = F.pad(freq_enc(w_l_camera_enc_b),[0,0,0,0,1,1,0,0])
					w_l_conditioning_ims.append([left_encoded_im])
					w_l_ff_refs.append([ray_downsampler(w_l_ff_ref.to(device))])
					w_l_ff_as.append([ray_downsampler(w_l_ff_a.to(device))])
					w_l_ff_bs.append([ray_downsampler(w_l_ff_b.to(device))])

					# right wrist
					if 'wr_focal_y' in el:
						wr_focal_y = el['wr_focal_y']
					else:
						wr_focal_y = el['focal_y']
					w_r_tform_ref = np.eye(4)
					w_r_tform_a_relative = np.asarray(el['transformation_right'])
					w_r_tform_b_relative = np.linalg.inv(np.asarray(el['transformation_right']))
					w_r_camera_enc_ref = rel_camera_ray_encoding(w_r_tform_ref,cli_args.image_size,wr_focal_y)
					w_r_camera_enc_a = rel_camera_ray_encoding(w_r_tform_a_relative,cli_args.image_size,wr_focal_y)
					w_r_camera_enc_b = rel_camera_ray_encoding(w_r_tform_b_relative,cli_args.image_size,wr_focal_y)
					w_r_camera_enc_ref = torch.Tensor(w_r_camera_enc_ref).unsqueeze(0)
					w_r_camera_enc_a = torch.Tensor(w_r_camera_enc_a).unsqueeze(0)
					w_r_camera_enc_b = torch.Tensor(w_r_camera_enc_b).unsqueeze(0)
					w_r_ff_ref = F.pad(freq_enc(w_r_camera_enc_ref),[0,0,0,0,1,1,0,0]) # pad, must be %4==0 for group norm
					w_r_ff_a = F.pad(freq_enc(w_r_camera_enc_a),[0,0,0,0,1,1,0,0])
					w_r_ff_b = F.pad(freq_enc(w_r_camera_enc_b),[0,0,0,0,1,1,0,0])
					w_r_conditioning_ims.append([right_encoded_im])
					w_r_ff_refs.append([ray_downsampler(w_r_ff_ref.to(device))])
					w_r_ff_as.append([ray_downsampler(w_r_ff_a.to(device))])
					w_r_ff_bs.append([ray_downsampler(w_r_ff_b.to(device))])

					if cli_args.gripper_state and 'gripper_data' in el:
						gripper_state.append(el['gripper_data'])
					else:
						gripper_state = None

				# sampling loop
				sampling_eps = 1e-5
				sampling_shape = (len(w_l_conditioning_ims), 4, 32, 32)
				w_l_x, w_r_x = score_sde.sde_bimanual.prior_sampling(sampling_shape)
				w_l_x = w_l_x.to(device)
				w_r_x = w_r_x.to(device)

				timesteps = torch.linspace(sde_bimanual.T, sampling_eps, sde_bimanual.N, device=device)
				if gripper_state is not None:
					gripper_state = torch.Tensor(np.array(gripper_state).astype(np.float32)).to(device)

				for i in tqdm(range(0,sde_bimanual.N)):
					t = timesteps[i]
					vec_t = torch.ones(sampling_shape[0], device=t.device) * t
					_, _, std = sde_bimanual.marginal_prob(w_l_x, w_r_x, vec_t)
					w_l_x, w_l_x_mean, w_r_x, w_r_x_mean = score_sde.reverse_diffusion_predictor(w_l_x, w_l_conditioning_ims, vec_t, w_l_ff_refs, w_l_ff_as, w_l_ff_bs, w_r_x, w_r_conditioning_ims, w_r_ff_refs, w_r_ff_as, w_r_ff_bs, gripper_state=gripper_state)
					w_l_x, w_l_x_mean, w_r_x, w_r_x_mean = score_sde.langevin_corrector(w_l_x, w_l_conditioning_ims, vec_t, w_l_ff_refs, w_l_ff_as, w_l_ff_bs, w_r_x, w_r_conditioning_ims, w_r_ff_refs, w_r_ff_as, w_r_ff_bs, gripper_state=gripper_state)

				w_l_decoded = vqgan.decode(w_l_x_mean)
				w_l_intermediate_sample = (w_l_decoded/2+0.5)
				w_l_intermediate_sample = torch.clip(w_l_intermediate_sample.permute(0,2,3,1).cpu()* 255., 0, 255).type(torch.uint8).numpy()
				w_l_outputs = [x['left_output'] for x in batch]
				pathlib.Path(os.path.dirname(w_l_outputs[0])).mkdir(parents=True, exist_ok=True)
				for out, im in zip(w_l_outputs, w_l_intermediate_sample):
					im_out = Image.fromarray(im)
					im_out.save(out)

				w_r_decoded = vqgan.decode(w_r_x_mean)
				w_r_intermediate_sample = (w_r_decoded/2+0.5)
				w_r_intermediate_sample = torch.clip(w_r_intermediate_sample.permute(0,2,3,1).cpu()* 255., 0, 255).type(torch.uint8).numpy()
				w_r_outputs = [x['right_output'] for x in batch]
				pathlib.Path(os.path.dirname(w_r_outputs[0])).mkdir(parents=True, exist_ok=True)
				for out, im in zip(w_r_outputs, w_r_intermediate_sample):
					im_out = Image.fromarray(im)
					im_out.save(out)
	else:
		# original implementation
		with torch.no_grad():
			for batch in batches:
				if cli_args.verbose:
					for el in batch:
						print(device, el['img'])
				conditioning_ims = []
				ff_refs = []
				ff_as = []
				ff_bs = []
				for el in batch:
					im = Image.open(el['img'])
					im = crop_like_dataset(im)
					
					if not os.path.exists(el['orig_img_copy_path']):
						cv2.imwrite(el['orig_img_copy_path'],cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
					im = im[:,:,:3].astype(np.float32).transpose(2,0,1)/127.5 - 1
					im = torch.Tensor(im).unsqueeze(0).to(device)
					encoded_im = None
					if not cli_args.stable_diffusion_encoder:
						encoded_im = vqgan.encode(im)
					else:
						encoded_im = vqgan.encode(im)[0]
					
					focal_y = el['focal_y']
					tform_ref = np.eye(4)
					tform_a_relative = np.asarray(el['transformation'])
					tform_b_relative = np.linalg.inv(np.asarray(el['transformation']))
					camera_enc_ref = rel_camera_ray_encoding(tform_ref,128,focal_y)
					camera_enc_a = rel_camera_ray_encoding(tform_a_relative,128,focal_y)
					camera_enc_b = rel_camera_ray_encoding(tform_b_relative,128,focal_y)
					camera_enc_ref = torch.Tensor(camera_enc_ref).unsqueeze(0)
					camera_enc_a = torch.Tensor(camera_enc_a).unsqueeze(0)
					camera_enc_b = torch.Tensor(camera_enc_b).unsqueeze(0)
					ff_ref = F.pad(freq_enc(camera_enc_ref),[0,0,0,0,1,1,0,0]) # pad, must be %4==0 for group norm
					ff_a = F.pad(freq_enc(camera_enc_a),[0,0,0,0,1,1,0,0])
					ff_b = F.pad(freq_enc(camera_enc_b),[0,0,0,0,1,1,0,0])
					conditioning_ims.append([encoded_im])
					ff_refs.append([ray_downsampler(ff_ref.to(device))])
					ff_as.append([ray_downsampler(ff_a.to(device))])
					ff_bs.append([ray_downsampler(ff_b.to(device))])

				# sampling loop
				sampling_shape = (len(conditioning_ims), 4, 32, 32)
				sampling_eps = 1e-5
				x = score_sde.sde.prior_sampling(sampling_shape).to(device)

				timesteps = torch.linspace(sde.T, sampling_eps, sde.N, device=device)
				for i in tqdm(range(0,sde.N)):
					t = timesteps[i]
					vec_t = torch.ones(sampling_shape[0], device=t.device) * t
					_, std = sde.marginal_prob(x, vec_t)
					x, x_mean = score_sde.reverse_diffusion_predictor(x,conditioning_ims,vec_t,ff_refs,ff_as,ff_bs)
					x, x_mean = score_sde.langevin_corrector(x,conditioning_ims,vec_t,ff_refs,ff_as,ff_bs)

				decoded = vqgan.decode(x_mean)
				intermediate_sample = (decoded/2+0.5)
				intermediate_sample = torch.clip(intermediate_sample.permute(0,2,3,1).cpu()* 255., 0, 255).type(torch.uint8).numpy()
				outputs = [x['output'] for x in batch]
				pathlib.Path(os.path.dirname(outputs[0])).mkdir(parents=True, exist_ok=True)
				for out, im in zip(outputs, intermediate_sample):
					im_out = Image.fromarray(im)
					im_out.save(out)


# Remove duplicates based on the chosen key
def remove_duplicates(list_of_dicts, key_to_compare):
	seen_values = set()
	unique_dicts = []
	for d in list_of_dicts:
		if d[key_to_compare] not in seen_values:
			unique_dicts.append(d)
			seen_values.add(d[key_to_compare])
	return unique_dicts

if __name__ == '__main__':
	import torch.multiprocessing as mp
	data_path = cli_args.config
	import json
	data = json.load(open(data_path))
	print("Total files: ", len(data))
	import os
	if global_sfm_method == 'orbslam_bimanual':
		data = [item for item in data if not os.path.isfile(item["left_output"])] + [item for item in data if not os.path.isfile(item["right_output"])]
		data = remove_duplicates(data, 'left_img')
	else:
		data = [item for item in data if not os.path.isfile(item["output"])]
	print("Remaining files: ", len(data))
	devices = [f'cuda:{i}' for i in cli_args.gpus]

	if cli_args.external_gpu:
		import torch
		print('CUDA device count: ', torch.cuda.device_count())
		os.environ["CUDA_VISIBLE_DEVICES"] = "1"
		device = f'cuda'
		inference(data, device)
	else:
		if len(devices) == 1:
			inference(data, devices[0])
		else:
			mp.set_start_method('spawn')
			datas = chunks_num(data,len(devices))
			ar = list(zip(datas,devices))
			with mp.Pool(len(devices)) as p:
				p.starmap(inference, ar)
