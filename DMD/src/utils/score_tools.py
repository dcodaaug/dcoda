import torch
import random
import pudb

class Score_sde_model(torch.nn.Module):
	def __init__(self,score_model,sde,ray_downsampler,rays_require_downsample=True,rays_as_list=False):
		super().__init__()
		self.score_model = score_model
		self.sde = sde
		self.rsde = sde.reverse(self.score,probability_flow=False)

		self.ray_downsampler = ray_downsampler
		self.rays_require_downsample = rays_require_downsample
		self.rays_as_list = rays_as_list

	def score(self,x,cond_im,t, ff_ref, ff_a, ff_b):
		if self.rays_as_list:
			d_ff_ref = [self.ray_downsampler(rays) for rays in ff_ref] if self.rays_require_downsample else ff_ref
			d_ff_a = [self.ray_downsampler(rays) for rays in ff_a] if self.rays_require_downsample else ff_a
			d_ff_b = [self.ray_downsampler(rays) for rays in ff_b] if self.rays_require_downsample else ff_b
		else:
			d_ff_ref = self.ray_downsampler(ff_ref) if self.rays_require_downsample else ff_ref
			d_ff_a = self.ray_downsampler(ff_a) if self.rays_require_downsample else ff_a
			d_ff_b = self.ray_downsampler(ff_b) if self.rays_require_downsample else ff_b
		_, std = self.sde.marginal_prob(torch.zeros_like(x), t)
		cond_std = torch.ones_like(std)*0.01 # assume conditioning image has minimal noise
		score_a, score_b = self.score_model(x, cond_im, std, cond_std, d_ff_ref, d_ff_a, d_ff_b) # ignore second score
		return score_a

	def forward_diffusion(self,x,t):
		z = torch.randn_like(x)
		mean, std = self.sde.marginal_prob(x, t)
		perturbed_data = mean + std[:, None, None, None] * z
		return perturbed_data, z, std

	def t_uniform(self,batch_size,device=None,eps=1e-5):
		# eps prevents sampling exactly 0
		t = torch.rand(batch_size, device=device) * (self.sde.T - eps) + eps
		return t

	def reverse_diffusion_predictor(self, x, cond_im, t, ff_ref, ff_a, ff_b):
		f, G = self.rsde.discretize(x, cond_im, t, ff_ref, ff_a, ff_b)
		z = torch.randn_like(x)
		x_mean = x - f
		x = x_mean + G[:, None, None, None] * z
		return x, x_mean

	def langevin_corrector(self, x, cond_im, t, ff_ref, ff_a, ff_b):
		sde = self.sde
		n_steps = 1
		target_snr = 0.075

		# specific to VESDE
		alpha = torch.ones_like(t)

		for i in range(n_steps):
			grad = self.score(x, cond_im, t, ff_ref, ff_a, ff_b)
			noise = torch.randn_like(x)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
			step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
			x_mean = x + step_size[:, None, None, None] * grad
			x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

		return x, x_mean

class Score_sde_bimanual_model(torch.nn.Module):
	def __init__(self,score_model,sde_bimanual,ray_downsampler,rays_require_downsample=True,rays_as_list=False):
		super().__init__()
		self.score_model = score_model
		self.sde_bimanual = sde_bimanual
		self.rsde_bimanual = sde_bimanual.reverse(self.score,probability_flow=False)

		self.ray_downsampler = ray_downsampler
		self.rays_require_downsample = rays_require_downsample
		self.rays_as_list = rays_as_list

	def score(self, w_l_perturbed_data, w_l_cond_im, w_l_t, w_l_ff_ref, w_l_ff_a, w_l_ff_b, w_r_perturbed_data, w_r_cond_im, w_r_t, w_r_ff_ref, w_r_ff_a, w_r_ff_b, gripper_state=None):
		if self.rays_as_list:
			w_l_d_ff_ref = [self.ray_downsampler(rays) for rays in w_l_ff_ref] if self.rays_require_downsample else w_l_ff_ref
			w_l_d_ff_a = [self.ray_downsampler(rays) for rays in w_l_ff_a] if self.rays_require_downsample else w_l_ff_a
			w_l_d_ff_b = [self.ray_downsampler(rays) for rays in w_l_ff_b] if self.rays_require_downsample else w_l_ff_b

			w_r_d_ff_ref = [self.ray_downsampler(rays) for rays in w_r_ff_ref] if self.rays_require_downsample else w_r_ff_ref
			w_r_d_ff_a = [self.ray_downsampler(rays) for rays in w_r_ff_a] if self.rays_require_downsample else w_r_ff_a
			w_r_d_ff_b = [self.ray_downsampler(rays) for rays in w_r_ff_b] if self.rays_require_downsample else w_r_ff_b
		else:
			w_l_d_ff_ref = self.ray_downsampler(w_l_ff_ref) if self.rays_require_downsample else w_l_ff_ref
			w_l_d_ff_a = self.ray_downsampler(w_l_ff_a) if self.rays_require_downsample else w_l_ff_a
			w_l_d_ff_b = self.ray_downsampler(w_l_ff_b) if self.rays_require_downsample else w_l_ff_b

			w_r_d_ff_ref = self.ray_downsampler(w_r_ff_ref) if self.rays_require_downsample else w_r_ff_ref
			w_r_d_ff_a = self.ray_downsampler(w_r_ff_a) if self.rays_require_downsample else w_r_ff_a
			w_r_d_ff_b = self.ray_downsampler(w_r_ff_b) if self.rays_require_downsample else w_r_ff_b

		_, _, std = self.sde_bimanual.marginal_prob(torch.zeros_like(w_l_perturbed_data), torch.zeros_like(w_r_perturbed_data), w_l_t)

		w_l_cond_std = torch.ones_like(std)*0.01 # assume conditioning image has minimal noise
		w_r_cond_std = torch.ones_like(std)*0.01 # assume conditioning image has minimal noise

		score_a, score_b, score_c, score_d = self.score_model(w_l_perturbed_data, w_l_cond_im, std, w_l_cond_std, w_l_d_ff_ref, w_l_d_ff_a, w_l_d_ff_b, w_r_perturbed_data, w_r_cond_im, std, w_r_cond_std, w_r_d_ff_ref, w_r_d_ff_a, w_r_d_ff_b, gripper_state=gripper_state) # ignore second and fourth scores
		return score_a, score_c

	def forward_diffusion(self, w_l_x, w_l_t, w_r_x, w_r_t):
		w_l_z = torch.randn_like(w_l_x)
		w_l_mean, w_r_mean, std = self.sde_bimanual.marginal_prob(w_l_x, w_r_x, w_l_t)
		w_l_perturbed_data = w_l_mean + std[:, None, None, None] * w_l_z

		w_r_z = torch.randn_like(w_r_x)
		w_r_perturbed_data = w_r_mean + std[:, None, None, None] * w_r_z
		return w_l_perturbed_data, w_l_z, std, w_r_perturbed_data, w_r_z, std

	def t_uniform(self, batch_size, device=None, eps=1e-5, arm='left'):
		# eps prevents sampling exactly 0
		w_l_t = torch.rand(batch_size, device=device) * (self.sde_bimanual.T - eps) + eps
		w_r_t = torch.rand(batch_size, device=device) * (self.sde_bimanual.T - eps) + eps
		return w_l_t, w_r_t

	def reverse_diffusion_predictor(self, w_l_x, w_l_conditioning_ims, vec_t, w_l_ff_refs, w_l_ff_as, w_l_ff_bs, w_r_x, w_r_conditioning_ims, w_r_ff_refs, w_r_ff_as, w_r_ff_bs, gripper_state=None):
		"""
		The function predicts the next step in the reverse diffusion process, refining noisy samples while incorporating stochastic noise.
		This is essential for generating diverse and high-quality samples in score-based models.
		"""
		# Computes reverse drift terms for left and right components, guiding the samples towards the data distribution.
		# Computes diffusion terms, controlling the stochasticity in the reverse process.
		w_l_f, w_l_G, w_r_f, w_r_G = self.rsde_bimanual.discretize(w_l_x, w_l_conditioning_ims, vec_t, w_l_ff_refs, w_l_ff_as, w_l_ff_bs, w_r_x, w_r_conditioning_ims, w_r_ff_refs, w_r_ff_as, w_r_ff_bs, gripper_state=gripper_state)
		w_l_z = torch.randn_like(w_l_x)
		w_l_x_mean = w_l_x - w_l_f
		w_l_x = w_l_x_mean + w_l_G[:, None, None, None] * w_l_z

		w_r_z = torch.randn_like(w_r_x)
		w_r_x_mean = w_r_x - w_r_f
		w_r_x = w_r_x_mean + w_r_G[:, None, None, None] * w_r_z
		return w_l_x, w_l_x_mean, w_r_x, w_r_x_mean

	def langevin_corrector(self, w_l_x, w_l_conditioning_ims, vec_t, w_l_ff_refs, w_l_ff_as, w_l_ff_bs, w_r_x, w_r_conditioning_ims, w_r_ff_refs, w_r_ff_as, w_r_ff_bs, gripper_state=None):
		"""
		The function performs Langevin Monte Carlo (LMC) sampling to correct and refine samples by 
		nudging them towards the high-probability regions of the target distribution. 
		This is particularly useful for improving sample quality in score-based generative models.
		"""
		sde = self.sde_bimanual
		n_steps = 1
		target_snr = 0.075

		# specific to VESDE
		alpha = torch.ones_like(vec_t)

		for i in range(n_steps):
			w_l_grad, w_r_grad = self.score(w_l_x, w_l_conditioning_ims, vec_t, w_l_ff_refs, w_l_ff_as, w_l_ff_bs, w_r_x, w_r_conditioning_ims, vec_t, w_r_ff_refs, w_r_ff_as, w_r_ff_bs, gripper_state=gripper_state)
			w_l_noise = torch.randn_like(w_l_x)
			w_l_grad_norm = torch.norm(w_l_grad.reshape(w_l_grad.shape[0], -1), dim=-1).mean()
			w_l_noise_norm = torch.norm(w_l_noise.reshape(w_l_noise.shape[0], -1), dim=-1).mean()
			# The step size is calculated to ensure the desired SNR is maintained, controlling the influence of the gradient and noise during the update.
			w_l_step_size = (target_snr * w_l_noise_norm / w_l_grad_norm) ** 2 * 2 * alpha
			# Refines the samples (w_l_x) by adding the gradient term scaled by the step size. This step moves the samples closer to high-probability regions of the data distribution.
			w_l_x_mean = w_l_x + w_l_step_size[:, None, None, None] * w_l_grad
			# Adds noise scaled by the step size to maintain stochasticity and explore the distribution effectively.
			w_l_x = w_l_x_mean + torch.sqrt(w_l_step_size * 2)[:, None, None, None] * w_l_noise

			w_r_noise = torch.randn_like(w_r_x)
			w_r_grad_norm = torch.norm(w_r_grad.reshape(w_r_grad.shape[0], -1), dim=-1).mean()
			w_r_noise_norm = torch.norm(w_r_noise.reshape(w_r_noise.shape[0], -1), dim=-1).mean()
			w_r_step_size = (target_snr * w_r_noise_norm / w_r_grad_norm) ** 2 * 2 * alpha
			w_r_x_mean = w_r_x + w_r_step_size[:, None, None, None] * w_r_grad
			w_r_x = w_r_x_mean + torch.sqrt(w_r_step_size * 2)[:, None, None, None] * w_r_noise

		return w_l_x, w_l_x_mean, w_r_x, w_r_x_mean

class Score_sde_monocular_model(torch.nn.Module):
	def __init__(self,score_model,sde):
		super().__init__()
		self.score_model = score_model
		self.sde = sde
		self.rsde = sde.reverse(self.score,probability_flow=False)

	def score(self,x,t):
		_, std = self.sde.marginal_prob(torch.zeros_like(x), t)
		score = self.score_model(x, std)
		return score

	def forward_diffusion(self,x,t):
		z = torch.randn_like(x)
		mean, std = self.sde.marginal_prob(x, t)
		perturbed_data = mean + std[:, None, None, None] * z
		return perturbed_data, z, std

	def t_uniform(self,batch_size,device=None,eps=1e-5):
		# eps prevents sampling exactly 0
		t = torch.rand(batch_size, device=device) * (self.sde.T - eps) + eps
		return t

	def reverse_diffusion_predictor(self, x, t):
		f, G = self.rsde.discretize(x, t)
		z = torch.randn_like(x)
		x_mean = x - f
		x = x_mean + G[:, None, None, None] * z
		return x, x_mean

	def langevin_corrector(self, x, t):
		sde = self.sde
		n_steps = 1
		target_snr = 0.075

		# specific to VESDE
		alpha = torch.ones_like(t)

		for i in range(n_steps):
			grad = self.score(x, t)
			noise = torch.randn_like(x)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
			step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
			x_mean = x + step_size[:, None, None, None] * grad
			x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

		return x, x_mean

class Score_modifier(torch.nn.Module):
	def __init__(self,model,max_batch_size):
		super().__init__()
		self.model = model 
		self.max_batch_size = max_batch_size

	def __call__(self,x,cond_ims,std,cond_std,ff_refs,ff_as,ff_bs):
		# assert len(ff_refs) == len(ff_as) == len(ff_bs) == len(cond_ims)

		# generate indices to batch data
		indices = []
		for b in range(len(ff_refs)):
			for d in range(len(ff_refs[b])):
				indices.append([b,d])

		# compute scores in batches
		independent_scores_a = [[] for _ in ff_refs]
		independent_scores_b = [[] for _ in ff_refs]
		for start in range(0,len(indices),self.max_batch_size):
			batch_indices = indices[start:start+self.max_batch_size]
			batch_x = torch.stack([x[b,...] for [b,_] in batch_indices],0)
			batch_cond_ims = torch.cat([cond_ims[b][n] for [b,n] in batch_indices],0)
			batch_ff_refs = torch.cat([ff_refs[b][n] for [b,n] in batch_indices],0)
			batch_ff_as = torch.cat([ff_as[b][n] for [b,n] in batch_indices],0)
			batch_ff_bs = torch.cat([ff_bs[b][n] for [b,n] in batch_indices],0)
			batch_std = torch.stack([std[b] for [b,_] in batch_indices],0)
			batch_cond_std = torch.stack([cond_std[b] for [b,_] in batch_indices],0)
			batch_score_a,batch_score_b = self.model(batch_x,batch_cond_ims,batch_std,batch_cond_std,batch_ff_refs,batch_ff_as,batch_ff_bs)
			for idx,[b,n] in enumerate(batch_indices): # unpack scores
				independent_scores_a[b].append(batch_score_a[idx,...])
				independent_scores_b[b].append(batch_score_b[idx,...])

		aggregated_score_a = [torch.stack(sss,0).mean(0) for sss in independent_scores_a]
		aggregated_score_b = [torch.stack(sss,0).mean(0) for sss in independent_scores_b]

		aggregated_score_a = torch.stack(aggregated_score_a,0)
		aggregated_score_b = torch.stack(aggregated_score_b,0)

		return aggregated_score_a,aggregated_score_b

	def train(self): self.model.train()
	def eval(self): self.model.eval()

class Score_modifier_bimanual(torch.nn.Module):
	def __init__(self,model,max_batch_size):
		super().__init__()
		self.model = model 
		self.max_batch_size = max_batch_size

	def __call__(self, w_l_perturbed_data, w_l_cond_ims, w_l_std, w_l_cond_std, w_l_ff_refs, w_l_ff_as, w_l_ff_bs, w_r_perturbed_data, w_r_cond_ims, w_r_std, w_r_cond_std, w_r_ff_refs, w_r_ff_as, w_r_ff_bs, gripper_state=None):
		# generate indices to batch data
		w_l_indices = []
		for b in range(len(w_l_ff_refs)):
			for d in range(len(w_l_ff_refs[b])):
				w_l_indices.append([b,d])

		w_r_indices = []
		for b in range(len(w_r_ff_refs)):
			for d in range(len(w_r_ff_refs[b])):
				w_r_indices.append([b,d])

		# compute scores in batches
		w_l_independent_scores_a = [[] for _ in w_l_ff_refs]
		w_l_independent_scores_b = [[] for _ in w_l_ff_refs]
		w_r_independent_scores_a = [[] for _ in w_r_ff_refs]
		w_r_independent_scores_b = [[] for _ in w_r_ff_refs]
		for start in range(0,len(w_l_indices),self.max_batch_size):
			w_l_batch_indices = w_l_indices[start:start+self.max_batch_size]
			w_l_batch_x = torch.stack([w_l_perturbed_data[b,...] for [b,_] in w_l_batch_indices],0)
			w_l_batch_cond_ims = torch.cat([w_l_cond_ims[b][n] for [b,n] in w_l_batch_indices],0)
			w_l_batch_ff_refs = torch.cat([w_l_ff_refs[b][n] for [b,n] in w_l_batch_indices],0)
			w_l_batch_ff_as = torch.cat([w_l_ff_as[b][n] for [b,n] in w_l_batch_indices],0)
			w_l_batch_ff_bs = torch.cat([w_l_ff_bs[b][n] for [b,n] in w_l_batch_indices],0)
			w_l_batch_std = torch.stack([w_l_std[b] for [b,_] in w_l_batch_indices],0)
			w_l_batch_cond_std = torch.stack([w_l_cond_std[b] for [b,_] in w_l_batch_indices],0)

			w_r_batch_indices = w_r_indices[start:start+self.max_batch_size]
			w_r_batch_x = torch.stack([w_r_perturbed_data[b,...] for [b,_] in w_r_batch_indices],0)
			w_r_batch_cond_ims = torch.cat([w_r_cond_ims[b][n] for [b,n] in w_r_batch_indices],0)
			w_r_batch_ff_refs = torch.cat([w_r_ff_refs[b][n] for [b,n] in w_r_batch_indices],0)
			w_r_batch_ff_as = torch.cat([w_r_ff_as[b][n] for [b,n] in w_r_batch_indices],0)
			w_r_batch_ff_bs = torch.cat([w_r_ff_bs[b][n] for [b,n] in w_r_batch_indices],0)
			w_r_batch_std = torch.stack([w_r_std[b] for [b,_] in w_r_batch_indices],0)
			w_r_batch_cond_std = torch.stack([w_r_cond_std[b] for [b,_] in w_r_batch_indices],0)

			batch_score_a, batch_score_b, batch_score_c, batch_score_d  = self.model(w_l_batch_x, w_l_batch_cond_ims, w_l_batch_std, w_l_batch_cond_std, w_l_batch_ff_refs, w_l_batch_ff_as, w_l_batch_ff_bs, w_r_batch_x, w_r_batch_cond_ims, w_r_batch_std, w_r_batch_cond_std, w_r_batch_ff_refs, w_r_batch_ff_as, w_r_batch_ff_bs, gripper_state=gripper_state)
			for idx,[b,n] in enumerate(w_l_batch_indices): # unpack scores
				w_l_independent_scores_a[b].append(batch_score_a[idx,...])
				w_l_independent_scores_b[b].append(batch_score_b[idx,...])
				w_r_independent_scores_a[b].append(batch_score_c[idx,...])
				w_r_independent_scores_b[b].append(batch_score_d[idx,...])

		w_l_aggregated_score_a = [torch.stack(sss,0).mean(0) for sss in w_l_independent_scores_a]
		w_l_aggregated_score_b = [torch.stack(sss,0).mean(0) for sss in w_l_independent_scores_b]
		w_r_aggregated_score_a = [torch.stack(sss,0).mean(0) for sss in w_r_independent_scores_a]
		w_r_aggregated_score_b = [torch.stack(sss,0).mean(0) for sss in w_r_independent_scores_b]

		w_l_aggregated_score_a = torch.stack(w_l_aggregated_score_a,0)
		w_l_aggregated_score_b = torch.stack(w_l_aggregated_score_b,0)
		w_r_aggregated_score_a = torch.stack(w_r_aggregated_score_a,0)
		w_r_aggregated_score_b = torch.stack(w_r_aggregated_score_b,0)

		return w_l_aggregated_score_a, w_l_aggregated_score_b, w_r_aggregated_score_a, w_r_aggregated_score_b

	def train(self): self.model.train()
	def eval(self): self.model.eval()

class Score_modifier_stochastic(torch.nn.Module):
	def __init__(self,model):
		super().__init__()
		self.model = model 

	def __call__(self,x,cond_ims,std,cond_std,ff_refs,ff_as,ff_bs):
		assert len(ff_refs) == len(ff_as) == len(ff_bs) == len(cond_ims)
		independent_scores_a = []
		independent_scores_b = []
		n_conditioning_views = len(ff_refs)

		n = random.choice(list(range(n_conditioning_views)))
		cond_im = cond_ims[n]
		score_a,score_b = self.model(x,cond_im,std,cond_std,ff_refs[n],ff_as[n],ff_bs[n])

		return score_a,score_b

	def train(self): self.model.train()
	def eval(self): self.model.eval()

class Score_modifier_stochastic_sanity(torch.nn.Module):
	def __init__(self,model):
		super().__init__()
		self.model = model 

	def __call__(self,x,cond_ims,std,cond_std,ff_refs,ff_as,ff_bs):
		assert len(ff_refs) == len(ff_as) == len(ff_bs) == len(cond_ims)
		independent_scores_a = []
		independent_scores_b = []
		n_conditioning_views = len(ff_refs)

		n = random.choice(list(range(n_conditioning_views)))
		# if n_conditioning_views == 1:
		#     n = 0
		# else:
		#     n = 1
		# n = 0
		cond_im = cond_ims[n]
		score_a,score_b = self.model(x,cond_im,std,cond_std,ff_refs[n],ff_as[n],ff_bs[n])

		return score_a,score_b

	def train(self): self.model.train()
	def eval(self): self.model.eval()
