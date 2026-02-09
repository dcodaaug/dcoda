sb_32_4_config = {
	'ckpt_path': "../instance-data/model.ckpt",
	'embed_dim': 4,
	# 'n_embed': 16384,
	'n_embed': 16384,
	'ddconfig': {
		'double_z': False,
		'z_channels': 4,
		'resolution': 256,
		'in_channels': 3,
		'out_ch': 3,
		'ch': 128,
		'ch_mult': [ 1,2,2,4],  # num_down = len(ch_mult)-1
		'num_res_blocks': 2,
		'attn_resolutions': [ 32 ],
		'dropout': 0.0,
	},
    'lossconfig': {
        "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
        "params": {
            "disc_conditional": False,
            "disc_in_channels": 3,
            "disc_num_layers": 2,
            "disc_start": 1,
            "disc_weight": 0.6,
            "codebook_weight": 1.0,
        }
    }
}

