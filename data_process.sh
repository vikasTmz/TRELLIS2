python data_toolkit/dump_mesh.py Fusion360 --root datasets/Fusion360;

python data_toolkit/dual_grid.py Fusion360 --root datasets/Fusion360 --resolution 256,512,1024;

python data_toolkit/encode_shape_latent.py --root datasets/Fusion360 --resolution 512;
python data_toolkit/encode_shape_latent.py --root datasets/Fusion360 --resolution 256;
python data_toolkit/encode_shape_latent.py --root datasets/Fusion360 --resolution 1024;

python data_toolkit/encode_ss_latent.py --root datasets/Fusion360 --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024 --resolution 64;
mv datasets/Fusion360/ss_latents/ss_enc_conv3d_16l8_fp16_64/ datasets/Fusion360/ss_latents/ss_enc_conv3d_16l8_fp16_64_1024;

python data_toolkit/encode_ss_latent.py --root datasets/Fusion360 --shape_latent_name shape_enc_next_dc_f16c32_fp16_512 --resolution 64;
mv datasets/Fusion360/ss_latents/ss_enc_conv3d_16l8_fp16_64/ datasets/Fusion360/ss_latents/ss_enc_conv3d_16l8_fp16_64_512;

python data_toolkit/encode_ss_latent.py --root datasets/Fusion360 --shape_latent_name shape_enc_next_dc_f16c32_fp16_256 --resolution 64;
mv datasets/Fusion360/ss_latents/ss_enc_conv3d_16l8_fp16_64/ datasets/Fusion360/ss_latents/ss_enc_conv3d_16l8_fp16_64_256;