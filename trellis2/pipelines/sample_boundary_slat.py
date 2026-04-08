"""
Boiler plate code for Generative classifier.

This model samples voxel occupancies; 1 if it is a boundary voxel, 0 otherwise.
We model p( o | z^surface ), where o is the occupancy and z^surface is the SLat (latent code and active voxels) of surface geometry.

The flow model generates logits Y_i and we get p_i = sigmoid(Y_i). Occupancy (or a binary mask) o_i = 1[p_i > τ]

coords^{boundary} = coords^{surface} [o == 1]
"""

flow_model = self.models_sparse_structure_flow

reso = flow_model.resolution
in_channels = flow_model.in_channels
noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
if self.low_vram:
    flow_model.to(self.device)
z_s = self.sparse_structure_sampler.sample(
    flow_model,
    noise,
    **cond,
    **sampler_params,
    verbose=True,
    tqdm_desc="Sampling sparse structure",
).samples
if self.low_vram:
    flow_model.cpu()

logging.info(f" \n\n sample_sparse_structure z_s shape: {z_s.shape}  \n\n")

# Decode sparse structure latent
decoder = self.models["sparse_structure_decoder"]
if self.low_vram:
    decoder.to(self.device)
decoded = decoder(z_s) > 0
if self.low_vram:
    decoder.cpu()
if resolution != decoded.shape[2]:
    ratio = decoded.shape[2] // resolution
    decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
