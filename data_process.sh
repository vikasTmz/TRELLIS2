python data_toolkit/dump_mesh.py AutoBrep_Dataset --root datasets/AutoBrep_Dataset;

python data_toolkit/dual_grid.py AutoBrep_Dataset --root datasets/AutoBrep_Dataset --resolution 256,512,1024;

python data_toolkit/encode_shape_latent.py --root datasets/AutoBrep_Dataset --resolution 512;
python data_toolkit/encode_shape_latent.py --root datasets/AutoBrep_Dataset --resolution 256;
python data_toolkit/encode_shape_latent.py --root datasets/AutoBrep_Dataset --resolution 1024;

python data_toolkit/encode_ss_latent.py --root datasets/AutoBrep_Dataset --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024 --resolution 64;
mv datasets/AutoBrep_Dataset/ss_latents/ss_enc_conv3d_16l8_fp16_64/ datasets/AutoBrep_Dataset/ss_latents/ss_enc_conv3d_16l8_fp16_64_1024;

python data_toolkit/encode_ss_latent.py --root datasets/AutoBrep_Dataset --shape_latent_name shape_enc_next_dc_f16c32_fp16_512 --resolution 64;
mv datasets/AutoBrep_Dataset/ss_latents/ss_enc_conv3d_16l8_fp16_64/ datasets/AutoBrep_Dataset/ss_latents/ss_enc_conv3d_16l8_fp16_64_512;

python data_toolkit/encode_ss_latent.py --root datasets/AutoBrep_Dataset --shape_latent_name shape_enc_next_dc_f16c32_fp16_256 --resolution 64;
mv datasets/AutoBrep_Dataset/ss_latents/ss_enc_conv3d_16l8_fp16_64/ datasets/AutoBrep_Dataset/ss_latents/ss_enc_conv3d_16l8_fp16_64_256;