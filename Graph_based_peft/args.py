import argparse

print('Parsing args')

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
parser.add_argument("--dataset", type=str, default="mrpc")
parser.add_argument("--task", type=str, default="mrpc")
parser.add_argument("--bs", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--head_lr", type=float, default=5e-3)
parser.add_argument("--fft_lr", type=float, default=1e-1)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warm_step", type=float, default=0.06)
parser.add_argument("--train_ratio", type=float, default=1)
parser.add_argument("--scale", type=float, default=100.)
parser.add_argument("--width", type=float, default=200.)
parser.add_argument("--fc", type=float, default=1.)
parser.add_argument("--share_entry", action= "store_true")
parser.add_argument("--set_bias", action= "store_true")
parser.add_argument("--seed", type=int, default=00000)
parser.add_argument("--entry_seed", type=int, default=2024)


parser.add_argument("--target_KK", type=int, default=20)
parser.add_argument("--target_ll", type=int, default=10000)
parser.add_argument("--init_warmup", type=int, default=1)
parser.add_argument("--final_warmup", type=int, default=2)
parser.add_argument("--mask_interval", type=int, default=100)
parser.add_argument("--beta1", type=float, default=0.85)
parser.add_argument("--beta2", type=float, default=0.85)
parser.add_argument("--MODE_SA", type=str, default="True")

  

parser.add_argument("--multi_heads", type=str, default="roberta.encoder.layer")
parser.add_argument("--multi_heads_fa", type=str, default="roberta.encoder")
parser.add_argument("--transform_type", type=str, default="HOSVD") 
args = parser.parse_args()

def get_args():
    return parser.parse_args()