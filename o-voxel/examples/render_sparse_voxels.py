#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


import torch
import json

# from trellis2.pipelines import Trellis2ImageTo3DPipeline
# from trellis2.modules.sparse import SparseTensor

from trellis2.utils.vae_helpers import *


def generate_combinations(
    ss_dir: Path, shape_dir: Path
) -> Iterator[Tuple[str, Path, Path]]:
    ss_map: Dict[str, Path] = {p.stem: p for p in ss_dir.glob("*.npz")}
    shape_map: Dict[str, Path] = {p.stem: p for p in shape_dir.glob("*.npz")}

    parquet_name2sha = {
        "sample_000037_edges": "a52932b08715905f6ba73e4062db7fad6a680c94816f6f4b7dac6b6e87510e06",
        "sample_000017_edges": "39fd82a3081869babc232ab479ce66fa6f2eb760170f9d943260eda6454e95e1",
        "sample_000098_surface": "457513428736d69e09eb965c28f2a67b2b0462234b15ad321d0012772a0884a5",
        "sample_000094_edges": "9a105690e3574e162402fc6289b7d19f70fe0ce11ab2bf9764368d55587384fe",
        "sample_000010_edges": "d9b4a0397b661e474bb65bf99d9419f8520d7c6254d2f67e35a4a38d680d36f3",
        "sample_000018_surface": "ef2d280c2d850701d618ea9d35803ea47668806b14ea09226542d59412b38dfb",
        "sample_000039_surface": "c3a427c4ec8c897ced04224295e3e937a9bb92cde812cb87280ce4567cd04430",
        "sample_000001_surface": "77851619e9011ebe003fffa059e9261573a6310a8a7a44de2a57a508fffdce4a",
        "sample_000059_edges": "ed525b232c4eec605a2fad1a17ee87a9918afd7645f089a1337b7dc9452db4bb",
        "sample_000085_edges": "97e84720d518e5c5287fa253ff02f545a94a1cc3c038ac819945fe3123e19a79",
        "sample_000034_edges": "7fd0646f08a2e096760836d87f89cdc44fa1bb0d641cb8f83382ac61ccd38c84",
        "sample_000096_edges": "a244a0090840d952507858c86bd4de455536d92378b5c3b40b6240aec33f5dbb",
        "sample_000087_surface": "733d026581dbeac37cd907929ff26fda1eb62c392ba40da9fba89d2db9c1c6f3",
        "sample_000069_edges": "9f758554846504759038cc55a9590463355251d9f04ef2b8f29249d13ce4f5aa",
        "sample_000007_edges": "951c2ec3dc9f659e88fdefc8b9e91cb71c39878b013143abf48807f311e37c4a",
        "sample_000070_edges": "5f50067fd6f33d3c93ad0a2a237a5b5dd57e2e50df68e66cd6c851fdbc33a434",
        "sample_000054_surface": "45762ebc4aa6aff8c341d4c6b9ae18d1773d5a629830721a9cdc213e025ab4d6",
        "sample_000023_surface": "c60fe040ec28e1a4649cf40adf3b13bd40c18bdc46fc78a8df1f02faebb4e60c",
        "sample_000033_edges": "97d3544122d252cfa880ae9bf18aa83e305770f13b3aacde68804bc4838768d3",
        "sample_000061_edges": "91e11026662fa67ab59b679187d8c5311e35fcd7421ccab132b74090d66d69e4",
        "sample_000077_surface": "21ce7ab5e687fead1774489f5ac76a5199a589e2507a25481ef0f7c19841955c",
        "sample_000099_surface": "1eadaeb4641da63318ecbfb54e05edf3782f0bf14afb1adb7cf2bf0c829f7ddd",
        "sample_000035_surface": "1ae23ac0a012d415edc1f6d026383af89735d8b162eccc1be6904abd44e91d0b",
        "sample_000049_edges": "80cedc5af0be6727777c86753f95663bbfab2eb00dfa96ead825efc2e49797c0",
        "sample_000062_edges": "ae0db608baf54be6a3b22365d7a317dffd651d6fe58d693eab936f6c785d20c7",
        "sample_000000_edges": "b35b849697aa3367ee2aae080ca8decbf3125a9c082c7b58d1f567a1986b3bfe",
        "sample_000026_surface": "3e9bfd383f9bbdb34270471f021b9daf7c213e7622b73b4e6c8aa923c6da1783",
        "sample_000040_edges": "c8a39a5116817d1d474a50a9f5eb8e7c92d31cb754a1e2dc243375887db20841",
        "sample_000003_edges": "9af8b4cbe14d1d26e648d1bbb57177cbe7ec3616df4a06ba006b9612644fd4c4",
        "sample_000069_surface": "d636947cdd1a69045e25a280b6109ade7a72e94984365156e36e9502dfd748d7",
        "sample_000005_surface": "e45027fbf966da2d5627298f7cd7c2eaa0d72271ace57a944cb2f345fb7a1f68",
        "sample_000061_surface": "4a985df25d4282ea94efdb9424a874d619e8d5e0ccd307157d77fe7245d60221",
        "sample_000060_surface": "30ce87d94060572c6cd0d323a9a7ce0c883344fc35310241a35468641535310d",
        "sample_000044_surface": "154774ba8303f183b212e8197188e2c07a42570300806cc5958d9d4613d79388",
        "sample_000035_edges": "4b13667d2e2da0dde8d433219ba1922f4fb873c41a4e6cbbe4d6c59df60ff6c2",
        "sample_000015_surface": "a9f12c54775eb23d72d2b78474fcb3a72e48965afb82823160160fd2d6763f71",
        "sample_000029_edges": "a32b902e118abbddb74fa49e75b6e88f8fd42f6f238b34d9661bec29d22bef31",
        "sample_000046_edges": "d1ff95c770b70a217090c5c9926134eaf5663b7732d70f44052b010b7ccf7a44",
        "sample_000094_surface": "a0453c2263e61664d9d6182f2a313a9609c32ad4ea81d237ecf4942c96d2a591",
        "sample_000071_surface": "0787f106c738a4802105eb71f9e13e5f394336767a15dad2a11fe23fd01a3f0e",
        "sample_000040_surface": "9184403b4dffce02c87f0b88d48446fbdb1d03324e23efa6be46848960f5e89a",
        "sample_000071_edges": "bb5db4e0c7b77ac6caac334bfb817d0afd9c98f7ee864b4ef7a04a21e62c8e1d",
        "sample_000028_edges": "64dd106c51d72c6c01a958614f7bdfc640a85436131be13570dcf2a2f3420640",
        "sample_000064_surface": "ed4165c96aeb8acc849d52cf49649f681e9335d9477fe021a4d05be36e3fa477",
        "sample_000091_surface": "7151d6ce71c1ed49f2ad8a14112030e55862af80464e9daee45ef82971f93a8b",
        "sample_000095_surface": "d0c61d756dcb5f35cb4bd238bba2b6aab55ad14c94baa5cb48362e6e2a190efa",
        "sample_000074_surface": "c6ed53f89492bda1cd8bbb266e7f2ace7edae6670c42fcfbeae7c228eb36168e",
        "sample_000056_surface": "227d9e069e21594bf000189ac20e4b26cc9b50ed1d3e86d84a069fd71c80c6de",
        "sample_000079_edges": "9e9b8410faf8b6f955fe6e386841adff77f2739a6448a8ea48ec237eb66e997d",
        "sample_000057_edges": "30402a27db3823f1d8c28c03733d62d04be8ce53e870c4ede80675587788a9f5",
        "sample_000043_edges": "1a0f2312d504a08c831d1ee467b7c4aa5a815da7d691a0cb5b3242151086a522",
        "sample_000072_surface": "a7488aaf918a772ad3b578f1d4162d9560c951464159e695a8d316fa1d49479e",
        "sample_000080_edges": "a68138e1e0a44ce5cebfcf7a35e4da9b30c788322cdd9ae970ca2452c66eeb01",
        "sample_000020_surface": "6d0f2bab38403364dc2deea3f093dd44e599ff1108cf740e4a0a1fbad9faccdd",
        "sample_000090_edges": "de844fa0532910d0dafb469c3b181d455e453304141ae05fdc0e288873c27e25",
        "sample_000031_edges": "c5ba78e4c6321db5ce5a8f9a44b920143ea9ff587c0168f58ec3961cbf85b259",
        "sample_000004_surface": "d54ff91869a68e2da10ec52ec8e71bb11749f16e6a5cd6272c84199f0582b1bf",
        "sample_000045_edges": "f6d8033d4f931eeb5a1c8dc266f12c6cde7ff96bda412009d3c0293a5e8d26c7",
        "sample_000083_surface": "4dcd53fa7be3f54b8ac45a42eb6be5860b9f78d3d972da1976adb87fe84bed47",
        "sample_000041_edges": "58bdac3630b7f07b8451e558e8dee781413b3d5a08cc525523593d797c55774a",
        "sample_000093_surface": "3550f626b4bd374437e74f9e4e2e5454d0db4005dacaa33de24d2a5fa21373fe",
        "sample_000048_surface": "f48e2b77625179b0bc914763c7411f87e743f7a382d8695993aaf1907f130b70",
        "sample_000086_edges": "b87a179604f8a970e14ee22516c5a68299c0a539834ec71d3436de029d196bdd",
        "sample_000038_edges": "f5973606fcb07198aa696b9e4c4dd9e09ef8185c40e08dec8f4a81e3dbb5b8bf",
        "sample_000000_surface": "691a9c9b9e9abf908f746c294b5f1bcb9c7d293cdebedb5ed687f1021c9f8fae",
        "sample_000015_edges": "7b1cc2b263096062b4fbfff213849b3916bcd0244142028b5972706f5427ffcd",
        "sample_000055_edges": "75c6bc9a1e9407496b36cbbb4a8ccb04506490ab03d224477cf1b196d5819728",
        "sample_000030_surface": "3bab9e0dbdee2c4885c067c6e370e1c1aa79c655d6acb8d616e67ea2a14b9994",
        "sample_000092_surface": "c78960d2ddaab53b703a570d49b000df7cc91df88173113f072a1891ee599acd",
        "sample_000058_edges": "9a3f8a317a67b12841e9ea27afc02d04c010a784ecaae94fc6b953246e995d99",
        "sample_000003_surface": "7a104553f4490bdbbc4cd0a5782bbc3bf6dd8bfb72a2deeb189be1e215bffd49",
        "sample_000068_surface": "d463cceb49c7c0586d88e208ae705310ef3f2013032393f0e38e44119d831a0c",
        "sample_000097_edges": "d8865c0ca0f9de6dc365995b4a00347fb86174df260caca7f8b0947a64c61d74",
        "sample_000050_surface": "b9966b54784ff0986d8bad8210f5a4cd5ff3d58835c467f37bcb26cc53f39a04",
        "sample_000066_surface": "6f691f0f461ff9c76d24b22a3faa1be4402352e57d22aa7a299d6c4863daef67",
        "sample_000062_surface": "913a98f6025d9a1453974838893a64b3ff9622c27e64a6f3c7e3ff722988dd0f",
        "sample_000039_edges": "1e2ffcd0cd7d4dc647903d0720b1dab4ee7fb9e922cdfbf2edb9a7fe6ad67b58",
        "sample_000066_edges": "06df6653beea27032fc492c94728649c2bb54159038ae10defa40d1aa7cb2853",
        "sample_000043_surface": "2b9fd301b8be65e53b7735a29f0640f0cff96f986417a5546056932ba916f3d4",
        "sample_000054_edges": "1efe189621eb53be0eb826e05f06b8dba43a284bb304e84bebb6d34773a5907a",
        "sample_000011_edges": "067a6b489a88280e92fba377cbeeaf9a81d65a252087b6223ab50f8d97fc1448",
        "sample_000007_surface": "083fc2bf7e3d9e9bb71130b6d741e84f49394f36723aca5d3e28b8986e58f73d",
        "sample_000013_edges": "5c40496ff4c10cb734ad2e4f69d932546e745f18adae6d959d143ca3f41ad8dd",
        "sample_000082_surface": "be28a2420af5cecb3a3ebff89b60be1403047660264cfd1ecae423eaaa1f7f07",
        "sample_000034_surface": "afa6f9d71cd38ca2494ff5417c8a35b3eb6d03ad8150c1c6d34c7e560aa4c7c0",
        "sample_000052_edges": "a303b443bdb4c2ddbdcd5af1c34d81bcff0f699b9e2260c1974f00d5ba996edc",
        "sample_000036_edges": "52ec576f036144ea79467f842db6abf58d3dbbe2d317bc5b89e8897dbf459f78",
        "sample_000059_surface": "636d44e7449aa8652476a063e7ef9b6472a56962691ac383379de28724821c17",
        "sample_000074_edges": "f73e5b915570c93ad86513096bef06e276d20103c857708025f5fc8918f3f9f1",
        "sample_000063_edges": "cf318060b55716c0852c277f470216e5a423afed6c0349c0aed7378a1edbecb5",
        "sample_000008_surface": "c55e7187df3a310e7f882eeecd597ca0a10570473b77168b4a4171416d63b6c0",
        "sample_000023_edges": "91519a0a37e6f8799c25edc5f4b4b42fe91bf6610d5f08ce9e7a01b5d4737de0",
        "sample_000049_surface": "01672a5ac730d6074fd01a6b946164a58036f8118941ac532d493829f208b7c1",
        "sample_000072_edges": "07d5a4f1e883d75ff48f3c14b8d150e7de113b6a8c1d3da0a3c9dacf16c2ab81",
        "sample_000024_edges": "dedbe7894c00ee80db9ed37ab3dd79b8f3b85b4c1d4e4158874e598978908b33",
        "sample_000082_edges": "d66daaa973dc8de6604b3e5c124403ce92bbd50c6fefffb1623dceb70a021d53",
        "sample_000051_edges": "94083d3da1c7642520097bd1cac5df0ec59d398cacc93d234c2e989f87442650",
        "sample_000077_edges": "76f38c0f7452e80af3f11b56a5d40c9fa211cc9977e5825baeccc8b1696258c1",
        "sample_000014_edges": "0b3b08b306950b954e362e0529ff08f9f4d83ed9d29787639fab33c31b055fdd",
        "sample_000022_surface": "6e0e6363b854168a005fc05d6fa5ae9f5acc54704f1abca7ec440bcbf6d8d839",
        "sample_000098_edges": "541465462ba9a7af9c874f4aa14bc3381ebd1ac4f59cf21a7439b114c001a1f7",
        "sample_000081_edges": "95133215be647d64f0244ae197ad1f713ccd603760294e49bc9cd533ccd57138",
        "sample_000084_edges": "5d11fdbf9fcd230442bfe3068f638409c068b6d95e6fb22c6fcd8fd0f2ac77be",
        "sample_000060_edges": "f32c2d11a036be87c304a5ac0df55585af38bbbd674a5d793f1dee982f13f9ac",
        "sample_000099_edges": "5e934b66e9d4df149f56f6a64398bac06b4a3ff4fd2d946adce126f809fb2cf1",
        "sample_000081_surface": "75282186cd7a4640436ab2c98075a5a7335029ee672316edb7d00b0f41435722",
        "sample_000033_surface": "9ed34ac6ca40c839fea2676defdd5b83cf3b37f9fd8e1cf00d664024457831e9",
        "sample_000076_surface": "804f4f0bf894ec826292a9b4e2ffb5992e0ab0752b9a0e0a18112c50756fe663",
        "sample_000012_surface": "857289bcbc57a5387b90aa552374b150caf4578aefa5cce5e0aa47ec67f8c240",
        "sample_000019_edges": "57a7ced2cfc8010204de5436a37b6a34a5ab1726622e65ec8a02e231f3b1dfc7",
        "sample_000006_edges": "46d920c8d9dfcdd49e1c95e2f181bfd929ea71a8ab84b061855d814ec4734a6a",
        "sample_000092_edges": "87ab590107255fc11be529e01c0a81b1f2fabc2d701998d55ee4130899511970",
        "sample_000030_edges": "a89bf0771de04818b3228917f21ad1ba02791b89fcb619818ea33ee06409aa4b",
        "sample_000063_surface": "7e3c10356f0409d0cf66865507fe674d52a93bce44fb31bc0898bb82d1aff49d",
        "sample_000026_edges": "c02488a24e87cae0365df54526e38a3c38ae54cf742a78f20389c3f392b072d2",
        "sample_000013_surface": "823175b48eefffb2b5bab9db1e63c3bf6a512c424876ece14ac7cb53c575e80d",
        "sample_000002_edges": "ce432c8d66d61a60dca2a50bcbc894f68a5527d4908ffb7683558baea7650465",
        "sample_000001_edges": "c482fbaff42b5fa580ddeaaec61aafd72267bb8a131ee72bc8ff91391d2936b7",
        "sample_000067_edges": "29200e6bf1954de2e3654a2e31af47f1c27ec4a60851b0091fb812cac7136975",
        "sample_000010_surface": "788c840ad3faedfba943d99de35bab5d35863d00e1dec63d4dd5a5bd7048c8b9",
        "sample_000004_edges": "138efcf0de3af696d13f504258a5e11704d4c1cf3647c1012c7c62432eb9ad6d",
        "sample_000008_edges": "e84b0b1b4bf54ea1c4b6a9c0c369db89bced4a404892fa0fc266ca4f35951e31",
        "sample_000052_surface": "5642a0a89e394aba6ef09b6d23a70a4cfa988f87a1256455deb28294c2e7df19",
        "sample_000067_surface": "c40a895bd71d1b7777f3ec090b493d6819770cb066d601731504ced7143b5976",
        "sample_000065_edges": "e94e32edb8e197ae763d3602c3a820132115054ab9b126a0894e113b2c30776e",
        "sample_000090_surface": "a5db0caa71cf2d13748cdc8ed2545eb6ad6f979bfd20ebb33845868ed6168032",
        "sample_000076_edges": "3e87f0657a04283b69ae6b905430d537428513f607af9bcbd7f8af9f4300285b",
        "sample_000095_edges": "18ee9c20d5db2b0a985a465268c091e17110823f8a2563c62c68028275d4f6cb",
        "sample_000058_surface": "aa855486a1a7004969dede2b38d1a0a8ab6d8d5abebe0bc0f5cdf6a7660f4ecb",
        "sample_000017_surface": "a25daa612755a655df802a5e75b36f9cb16cbff5760ec84528503f29ff191b61",
        "sample_000027_edges": "49b163e4a8483b7b8b6a441b14991b27db999ef6593070cc10b97b7028f8c952",
        "sample_000032_edges": "3dc63f8f368ea08adb73253e8db95f495e6b2fa2212246ce50f8977577f2ba95",
        "sample_000057_surface": "b86d6913d3762c0aa6bb86e0b27f92caefce01838fa9e6cb41b4cba4853a0fea",
        "sample_000048_edges": "f53560843a747a6f5a15f2596317b116c839ce345cfb86bdc3872e1909fa7bf4",
        "sample_000055_surface": "ada643f3302351e55010f3ba32f0f1dca030bb600389e9801ba28df847c10c4b",
        "sample_000042_edges": "95402fe4355b64c90c9f287d98903d6703f7f11156767702b884045cbfd90c07",
        "sample_000044_edges": "5d4cdee837f9aec51dbdf0a979f5b61076e666d45361f9afc794eb64fc66d3e0",
        "sample_000021_surface": "aa935aff918fff8a3b67b8ae0ac30fe8191e59ed6d8d6d3239acabbc28fd06a4",
        "sample_000056_edges": "fa26463cad3be280e1f7e1df67c1adadf001367cb35e57b0062df7160ee6bf6a",
        "sample_000046_surface": "a4d3e0bd7290692732db35f508d305fa3fccc1662145f330d93f560f11ff2ea1",
        "sample_000097_surface": "50243e8733972b04b137eeef93de09fcdb56646fa32b08635130e4ed0ec5d7da",
        "sample_000089_surface": "fb9192d28ac9527dba522be9a2b65c43dd9c7eb041d7b58767b282e073be2bc8",
        "sample_000020_edges": "0075fbc491d9d2b81f2b5a748927ca813d1cc789196d5eeb1bca0f8e585b1ccd",
        "sample_000075_edges": "6b41712993f66b8e50a59cfbee6abe9c417a3e33951abfa38d9b52d81afe18b8",
        "sample_000014_surface": "a85de2566ff4ae864163dc457cec3562f5e41217eb5427f1f1d1b6ca058d9429",
        "sample_000091_edges": "e2a34740036af1fce43e025058d7f52bd8cdb1ff0c3ac259a038832189d9e95b",
        "sample_000050_edges": "a46b867aa3bea78130237df2a683da853d0eccfd46bc9b3dd9e22e42496f0c06",
        "sample_000009_surface": "97dbc7e2ba9898e9a556e4107f0546f8b2efaff5b5d055cb93af08d6849d92a4",
        "sample_000038_surface": "48f904462de0b14928b9b490271fe376f66cfb4ac949f57546d025563b1c731b",
        "sample_000093_edges": "3bbab4766486040ede93a666f4d39a1f44a1030cc6aad41693e66b79132a733f",
        "sample_000009_edges": "c253512b7b4e8e9df6f63bedfed2f4a962a4f14a8953bd2aa566690037a9a648",
        "sample_000019_surface": "5d2e0815e9724ccc526caa3d91cce1f6039e9c1b08ac80436e08d6c3e5ce1a0e",
        "sample_000025_edges": "2717972c3190101c3b252123e8afba5dc4c689b1cef098b3c5bcab3d53061a6e",
        "sample_000011_surface": "37ab5ee463a6139c16c19c3b2f40752fc7cb3c635e121bbbca05757faf99223c",
        "sample_000078_edges": "69ed869795519f412929acbda0e187fcbc0399858e2bb3ae3726c142d3f89da7",
        "sample_000016_surface": "3017df9867b98c6f1915ee3056c35c7c2b5f45471e112019df34cc840757c1be",
        "sample_000045_surface": "59d05226fc4dd87eaf4ee72e208514fe25ec7c8e9ccfb4a379c0a414fcec73f3",
        "sample_000002_surface": "2d01184ebf202e965629d137d2e2761d013987bc4df808b435b71096e8919e5c",
        "sample_000024_surface": "b12445297933c434df4acac45309bc2ea9fa4c58f49500e36c9289f790a2ef95",
        "sample_000047_edges": "465a1183e5a810747b985d81dd0f6c73be7ba51c3617f4488a109aa04e95be28",
        "sample_000100_edges": "97370575ca66feb2cf5a5a775f7fc670f911ca4e4fd3c782eab32e78440fb575",
        "sample_000080_surface": "d318c70b0d4d7a84de8e69303bf1048f74db7246653b56333790f3b6b57be763",
        "sample_000065_surface": "1bf0a012cc20ca508794ef1f8126a29ff7dea890963b890561e7cb7042742b80",
        "sample_000031_surface": "56931a28a40015480c8e2d2871f1f7ba02199a6ed4847a4fb074eb7dcf9b499b",
        "sample_000068_edges": "5a0cccf383849280a92d4c485a1d9b0d4c7049a66695e9d63339ec4bfe1a48c3",
        "sample_000078_surface": "9402ea897175d6ec7f06c060e24ea377a5907023ac40a1e9b8572c973bb505af",
        "sample_000036_surface": "16230f668357e1fb35c0fdfb2de0424e5ab10f47c2dd57df935f93cdf1c0a154",
        "sample_000096_surface": "ed93a27b96f767a682ba61bdd8908f646370ad6ad9d4e90bc0cbcb771898e8df",
        "sample_000022_edges": "f506427332390765d6f458b9ae97dfae1a209f16b55686827f43d6eebe170e0c",
        "sample_000047_surface": "6a7c5aa393fe6f640cd86c0b711eb56961848a12faa88472844aab4fdd129766",
        "sample_000018_edges": "5d1d4a2705993116b95ac2a6e0548b51400630a5039ca038eaebddab690393e1",
        "sample_000042_surface": "1b8570c530c36293f810ea914edba94a3c193349569443657ed9cf5d515ff8d9",
        "sample_000073_surface": "0e43469b70f1136550c7f51e5d6e50dbf18d761911b5d49386e6bf2f27613ac5",
        "sample_000053_surface": "28b14662a408f11a930d142bf0c5b9a126c735fd04fc3b9b3923e9575bd0849d",
        "sample_000073_edges": "4a2ac6a92786a9d45843301a5acab97be44e90f46e35d55e354d7f6c59178f38",
        "sample_000088_surface": "85c2348bc65270d3b931df6b460d4f8a9135c208b36abe3e2b27ab7b7eb7dc70",
        "sample_000051_surface": "949d47eb8de1b89cd5015597941a8f5950b833e51e886f830f959a8785f615ef",
        "sample_000088_edges": "0d1761dcdfc874ed68135b359464002375287b0993f78c2fc3655e85f475d24c",
        "sample_000037_surface": "ccb0c797fd56cfe5866883b8e697813fd6935ce3d2ac46889408b6178fcb48b7",
        "sample_000085_surface": "14b181f523e34a5ddeb730593a4a98414a54e77e1cb1be05ef272b01ea4c77cd",
        "sample_000041_surface": "0fd392e3d8dc09967c14f6ec1a3f14f3f646ae68061a1c217810d2e6b97fc81f",
        "sample_000086_surface": "c904246ce9f2c122eea422a989c332d64e092a15055ee53ee5816564dd1a690c",
        "sample_000005_edges": "a59b5cfeb11f62c8e9e7c3cc8dc4efab7c5cf647e0a1a4ddfd7725db551987fa",
        "sample_000079_surface": "b3183d2df76f74223a49494f2d76447e68abae4ea3ac61ee1348d405d2fe8e17",
        "sample_000027_surface": "6451ddf140150641732f91f956af9e88143a91658524c6fd906a266b1c755dd9",
        "sample_000089_edges": "193a16ba49429c7615b03a6beb18a08cd783be5f701a6e67151ec50bbbf6df1c",
        "sample_000075_surface": "12cd9889d04d030f2f99c61832cb9ab0b5434e0678f6a45526d576dca3481ad0",
        "sample_000025_surface": "b608355dee237d86a4679b13912639d629b76ea669b18ca25a72ed3ab00659f7",
        "sample_000029_surface": "cfb7c19642ae0cfd84efa433297c6f9564b827118f1c56cfcf1022b8afd7b441",
        "sample_000064_edges": "562efdf8047d6f4b48e026ba2c1e9ebe6bd2baedc00bc47ff9603ec424d4f8bd",
        "sample_000021_edges": "e2e1671215e3805d8d4c27b43ad933bd771c82ba70466f50dd1cf367d1e6848b",
        "sample_000012_edges": "92cc6588102fc455b99810a4426a51083b01c5d8b4bfa7d17778c302180f930b",
        "sample_000028_surface": "c46ff5e881d3334890cb727623d0236854d094746bdfed086bf30d3444dbf9ff",
        "sample_000070_surface": "295d31a14991fd2d19fa3226ef89da810708ff1bbcc55c1ca7035518465dcd8a",
        "sample_000087_edges": "c875ebf804ec0d69f13e429f520bc1a40dfab7a714b2a47d0dcb774866714803",
        "sample_000006_surface": "8dd036b9410184a12db6241f81677f9d111eadd1af18fef62e98ba6b01a669d1",
        "sample_000016_edges": "a651d22d951585867bbb3e377c6c8d293c942a1b03867682d8f294c8cd493529",
        "sample_000032_surface": "6d930f0e947bc3e72273184cdabded6f4ead011d12d6724d6662e688612ad0da",
        "sample_000084_surface": "718738843ef48faddc3e1ecdfd2a8d1e0a3ded88573dc9861cf34e169a9c37a8",
        "sample_000053_edges": "a2be35cda47a0a4804fbe7e124363a3103d1b120ebdec4c7937e424984251287",
        "sample_000083_edges": "65a1778322ec08479fe4891eb47273963de3fe3188656a0dba31e1c7d2a6f16a",
    }

    for i in range(100):
        surface_name = f"sample_{i:06d}_surface"
        edge_name = f"sample_{i:06d}_edges"

        if (
            parquet_name2sha[surface_name] in shape_map
            and parquet_name2sha[edge_name] in shape_map
        ):
            yield f"sample_{i:06d}", shape_map[
                parquet_name2sha[surface_name]
            ], shape_map[parquet_name2sha[edge_name]]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

"""
python VAEdecoding.py  --dataset_root datasets/AutoBrep_Dataset --out_dir datasets/AutoBrep_Dataset/decode_shapes/exp1  --low_vram --use_ss_decoder --ss_target_res 64 --pool_on_cpu --decode_resolutions 1024 512 256 --decimate_faces 16777216 --dtype fp32
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("hf_token", type=str)
    parser.add_argument(
        "--dataset_root", type=Path, default=Path("datasets/AutoBrep_Dataset")
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--model_id", type=str, default="microsoft/TRELLIS.2-4B")
    parser.add_argument(
        "--low_vram", action="store_true", help="Enable TRELLIS.2 low_vram mode."
    )
    parser.add_argument(
        "--use_ss_decoder",
        action="store_true",
        help="Use sparse_structure_decoder to compute coords.",
    )
    parser.add_argument(
        "--ss_target_res",
        type=int,
        default=64,
        help="Target resolution for SS decoder coords.",
    )
    parser.add_argument(
        "--pool_on_cpu",
        action="store_true",
        help="Pool SS occupancy on CPU to save VRAM.",
    )
    parser.add_argument(
        "--decode_resolutions", type=int, nargs="+", default=[1024, 512, 256]
    )
    parser.add_argument(
        "--decimate_faces",
        type=int,
        default=16777216,
        help="If >0, simplify mesh to this face count.",
    )
    parser.add_argument(
        "--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"]
    )
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script expects a CUDA GPU.")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    compute_dtype = dtype_map[args.dtype]

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Create context for mixed precision
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if compute_dtype != torch.float32
        else nullcontext()
    )

    # Run Inference
    with torch.inference_mode(), autocast_ctx:
        for res in [1024, 512, 256]:
            # Construct paths based on resolution
            ss_dir = (
                args.dataset_root / "ss_latents" / f"ss_enc_conv3d_16l8_fp16_64_{res}"
            )
            shape_dir = (
                args.dataset_root
                / "shape_latents"
                / f"shape_enc_next_dc_f16c32_fp16_{res}"
            )

            if not ss_dir.exists() or not shape_dir.exists():
                LOGGER.warning(
                    f"Missing dirs for res={res}: \n  {ss_dir} \n  {shape_dir}"
                )
                continue

            LOGGER.info(
                f"Decoding resolution={res} from:\n  ss: {ss_dir}\n  shape: {shape_dir}"
            )

            # Process files
            exp_combinations = generate_combinations(ss_dir, shape_dir)

            for (
                key,
                surface_path,
                boundary_path,
            ) in exp_combinations:
                torch.cuda.reset_peak_memory_stats()

                LOGGER.info(f"Processing: {key}")

                # 1. Load latents (CPU)
                _, surf_coords_np = load_shape_latent_npz(surface_path)
                _, bound_coords_np = load_shape_latent_npz(boundary_path)

                LOGGER.info(
                    f"Loaded latents:  surf_enc_coords = {surf_coords_np.shape}, bound_enc_coords = {bound_coords_np.shape}"
                )

                surf_coords_np = surf_coords_np[:, 1:4]
                bound_coords_np = bound_coords_np[:, 1:4]

                # take intersection of surf_coords_np and bound_coords_np

                surf_union_bound_coords = np.array(
                    list(
                        set(map(tuple, surf_coords_np))
                        & set(map(tuple, bound_coords_np))
                    )
                )

                surf_minus_bound_coords = np.array(
                    list(
                        set(map(tuple, surf_coords_np))
                        - set(map(tuple, surf_union_bound_coords))
                    )
                )

                surf_and_bound_coords_1 = (
                    np.concatenate(
                        [
                            surf_minus_bound_coords,
                            bound_coords_np,
                        ],
                        axis=0,
                    )
                    if len(surf_minus_bound_coords) > 0
                    else bound_coords_np
                )

                surf_and_bound_coords_2 = (
                    np.concatenate(
                        [
                            surf_minus_bound_coords,
                            surf_union_bound_coords,
                        ],
                        axis=0,
                    )
                    if len(surf_minus_bound_coords) > 0
                    else surf_union_bound_coords
                )

                out_name = f"{key}_{res}" + (
                    "_from_dec-coords" if args.use_ss_decoder else "_from_enc-coords"
                )
                toCombine = True
                images = []
                out_path = None
                for coords, suffix, surf_idx, bound_idx in [
                    [
                        surf_coords_np,
                        "surf",
                        np.arange(len(surf_coords_np)),
                        None,
                    ],
                    [bound_coords_np, "bound", None, np.arange(len(bound_coords_np))],
                    [
                        surf_and_bound_coords_1,
                        "comb",
                        np.arange(len(surf_minus_bound_coords)),
                        len(surf_minus_bound_coords) + np.arange(len(bound_coords_np)),
                    ],
                    [
                        surf_union_bound_coords,
                        "bound_union",
                        None,
                        np.arange(len(surf_union_bound_coords)),
                    ],
                    [
                        surf_and_bound_coords_2,
                        "comb",
                        np.arange(len(surf_minus_bound_coords)),
                        len(surf_minus_bound_coords)
                        + np.arange(len(surf_union_bound_coords)),
                    ],
                ]:
                    if not toCombine:
                        out_path = args.out_dir / f"{out_name}_{suffix}.jpg"
                    img = render_voxels_pyvista(
                        coords,
                        surf_idx=surf_idx,
                        bound_idx=bound_idx,
                        out_path=out_path,
                    )
                    if toCombine:
                        images.append(img)

                if toCombine:
                    gap = 20
                    total_width = sum(img.width for img in images) + gap * (
                        len(images) - 1
                    )
                    max_height = max(img.height for img in images)
                    combined = Image.new("RGB", (total_width, max_height), "white")
                    x = 0
                    for img in images:
                        combined.paste(img, (x, 0))
                        x += img.width + gap
                    combined.save(args.out_dir / f"{out_name}.jpg")

                LOGGER.info(f"Wrote {out_name}")

    LOGGER.info("Done.")


# Python <3.10 compatibility helper
class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
