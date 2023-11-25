The paper is under review, and the complete code will be released later

## LeadDisFlow
Discovery of EP4 antagonists through image-guided deep learning workflow
&nbsp;

## Create conda env 
````
conda env create -f environment.yaml
````

## Prepare datasets for AnDisFlow
Download datasets from "Data availability".

&nbsp;


## First Step --> AnDisFlow-G
### 🌱Train the AnDisFlow-G

**Slice the EP4 datasets for train**
````
python slice_data2fragmentes.py --input_path your_ligands_file.csv --output_path your_ligands_file_fragments.csv
````

**Construct sliced datasets to muti-smi files**
````
python fragments2mutiSmi.py --input_path your_ligands_file_fragments.csv --output_path datasets/
````

**Traning empty model from one sliced smi files**
````
python create_emptyModel.py ---input_path datasets/one.smi --output_path datasets/model.empty
````

**Begin training**
````
python train_model.py --input_path datasets/model.empty --output_path datasets/model.trained -s datasets/train_fragments/
````


### 🪴Inference from the AnDisFlow-G

**Slice the EP4 datasets for inference**
````
python slice_data2fragmentes.py --input_path your_inference_file.csv --output_path your_inference_file_fragments.csv
````
**Begin inference**
````
python sample_scaffolds.py -m data_EP4/EP4_decorator/models/model.trained.90 -i data_EP4/EP4_decorator/CN109836434B_slice.smi -o data_EP4/generated_and_screen/CN109836434B_generated.csv -r 64 -n 64 -d multi --of csv 
````


&nbsp;

## Second Step --> AnDisFlow-S
### 🍏Train the AnDisFlow-S

````
cd Screen
````

Use the data for screen in folder(DataForScreen), and download pre-trained model and push it into the folder ckpts/
link: <https://drive.google.com/file/d/1wQfby8JIhgo3DxPvFeHXPc14wS-b4KB5/view>

Finetune:
````
python finetune.py --gpu ${gpu_no} \
                   --save_finetune_ckpt ${save_finetune_ckpt} \
                   --log_dir ${log_dir} \
                   --dataroot ${dataroot} \
                   --dataset ${dataset} \
                   --task_type ${task_type} \
                   --resume ${resume} \
                   --image_aug \
                   --lr ${lr} \
                   --batch ${batch} \
                   --epochs ${epoch}
````


More info about Imagemol, you can reference: <https://github.com/HongxinXiang/ImageMol>



