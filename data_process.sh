python data_toolkit/dump_mesh.py Thingi10K --root datasets/Thingi10K;

python data_toolkit/dual_grid.py Thingi10K --root datasets/Thingi10K --resolution 256,512,1024;

python data_toolkit/encode_shape_latent.py --root datasets/Thingi10K --resolution 512;
python data_toolkit/encode_shape_latent.py --root datasets/Thingi10K --resolution 256;
python data_toolkit/encode_shape_latent.py --root datasets/Thingi10K --resolution 1024;

python data_toolkit/encode_ss_latent.py --root datasets/Thingi10K --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024 --resolution 64;
mv datasets/Thingi10K/ss_latents/ss_enc_conv3d_16l8_fp16_64/ datasets/Thingi10K/ss_latents/ss_enc_conv3d_16l8_fp16_64_1024;

python data_toolkit/encode_ss_latent.py --root datasets/Thingi10K --shape_latent_name shape_enc_next_dc_f16c32_fp16_512 --resolution 64;
mv datasets/Thingi10K/ss_latents/ss_enc_conv3d_16l8_fp16_64/ datasets/Thingi10K/ss_latents/ss_enc_conv3d_16l8_fp16_64_512;

python data_toolkit/encode_ss_latent.py --root datasets/Thingi10K --shape_latent_name shape_enc_next_dc_f16c32_fp16_256 --resolution 64;
mv datasets/Thingi10K/ss_latents/ss_enc_conv3d_16l8_fp16_64/ datasets/Thingi10K/ss_latents/ss_enc_conv3d_16l8_fp16_64_256;