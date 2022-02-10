import torch
from transformers import BertTokenizer
from PIL import Image
import argparse
import torchvision.transforms as transforms
import os
import numpy as np
import torchvision as tv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

MAX_DIM = 299

def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',default='')
parser.add_argument('--result_dir',default='')
args = parser.parse_args()
print(args)

img_lists  = os.listdir(args.img_dir)

for filename in img_lists:

    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    max_position_embeddings = 128

    image_path =  os.path.join(args.img_dir, filename)
    print(f'img path: {image_path}')
    
    image = Image.open(image_path)
    # image.show()
    image = val_transform(image)
    # print(image.shape)
    image = image.unsqueeze(0)

    caption, cap_mask = create_caption_and_mask(start_token, max_position_embeddings)

    features = []
    atten = []

    def hook(module, input, output):
        features.append(output[1].clone().detach())

    handle = model.transformer.decoder.layers[5].multihead_attn.register_forward_hook(hook)

    model.eval()

    for i in range(max_position_embeddings - 1):
        # print(i)
        predictions = model(image, caption, cap_mask)
        # print(predictions.shape)
        # print(features[-1].shape)
        tmp = features[-1]
        # print(tmp.shape)
        atten.append(tmp[:, i, :])

        predictions = predictions[:, i, :]
        # print(predictions.shape)
        predicted_id = torch.argmax(predictions, axis=-1)

        # print(predicted_id.shape)
        
        if predicted_id[0] == 102:
            # return caption
            break

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
        
    output = caption
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(result.capitalize())

    atten = torch.cat(atten)
    # print(atten.shape)
    result = result.split()
    # print(len(result))
    origin_img = mpimg.imread(image_path)
    # print(origin_img.shape)
    im_size =  max(origin_img.shape[0],origin_img.shape[1])
    # plt.figure(figsize=(32, 16))
    n_col = 5 # math.ceil(len(result)/2)
    # print(n_col)
    fig, ax = plt.subplots(3, n_col, figsize=(15,10))

    ax[0,0].imshow(origin_img)
    ax[0,0].axis('off')
    ax[0,0].set_title('<start>')

    for idx, word in enumerate(result):

        # print(word)
        img = atten[idx]
        # print('weight shape ',img.shape)
        embed_size = int (int( list(img.size())[0] ) / 19 )
        f = torch.reshape(img,(embed_size,19))
        tfm = transforms.Compose([
            transforms.Resize((im_size,im_size))
        ])
        im = tfm(f.unsqueeze(0))
        im = im.squeeze()

        if idx + 1 < n_col:
            x_axis = 0
        elif idx + 1 < n_col * 2 :
            x_axis = 1
        else:
            x_axis = 2

        ax[x_axis,idx+1 - x_axis*n_col].imshow(origin_img)
        ax[x_axis,idx+1 - x_axis*n_col].imshow(im,alpha=0.8)
        ax[x_axis,idx+1 - x_axis*n_col].axis('off')
        ax[x_axis,idx+1 - x_axis*n_col].set_title(word)

        if idx == len(result) -1 :
            img = atten[idx+1]
            embed_size = int (int( list(img.size())[0] ) / 19 )
            f = torch.reshape(img,(embed_size,19))
            tfm = transforms.Compose([
                transforms.Resize((im_size,im_size))
            ])
            im = tfm(f.unsqueeze(0))
            im = im.squeeze()
            
            ax[2,4].imshow(origin_img)
            ax[2,4].imshow(im,alpha=0.8)
            ax[2,4].axis('off')
            ax[2,4].set_title('<end>')

    save_path = os.path.join(args.result_dir,f'{filename[:-4]}.png')
    plt.savefig(save_path)