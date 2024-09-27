import gradio as gr
import os
from pathlib import Path
import sys
from PIL import Image
from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError
import time

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

category_dict = ['upperbody', 'lowerbody', 'dress']

example_path = os.path.join(os.path.dirname(__file__), 'examples')
model_hd = os.path.join(example_path, 'model/model_1.png')
garment_hd = os.path.join(example_path, 'garment/03244_00.jpg')
model_dc = os.path.join(example_path, 'model/model_8.png')
garment_dc = os.path.join(example_path, 'garment/048554_1.jpg')

client = Client("jsoncm/OOTDiffusion", hf_token="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")

def process_images(result):
    processed_images = []
    for item in result:
        if isinstance(item, dict) and 'image' in item:
            img_path = item['image']
            if os.path.isfile(img_path):
                img = Image.open(img_path)
                processed_images.append(img)
            else:
                print(f"警告: 图像文件不存在: {img_path}")
        elif isinstance(item, (str, Path)):
            if os.path.isfile(item):
                img = Image.open(item)
                processed_images.append(img)
            else:
                print(f"警告: 图像文件不存在: {item}")
        else:
            print(f"警告: 无法处理的项目类型: {type(item)}")
    return processed_images

def process_with_retry(func, *args, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            result = func(*args)
            if result is not None:
                return result
            else:
                print(f"尝试 {attempt + 1} 失败，等待 {delay} 秒后重试...")
                time.sleep(delay)
        except AppError as e:
            print(f"远程应用程序错误: {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                raise
    return None

def process_hd(vton_img, garm_img, n_samples, n_steps, image_scale, seed):
    print(f"发送到 API 的参数: vton_img={vton_img}, garm_img={garm_img}, n_samples={n_samples}, n_steps={n_steps}, image_scale={image_scale}, seed={seed}")
    category = 0 # 0:upperbody; 1:lowerbody; 2:dress
    def _process():
        result = client.predict(
            vton_img=handle_file(vton_img),
            garm_img=handle_file(garm_img),
            n_samples=n_samples,
            n_steps=n_steps,
            category=category_dict[category],
            image_scale=image_scale,
            seed=seed,
            api_name="/process_hd"
        )
        print("API返回结果 (HD):", result)
        return process_images(result) if result else None

    result = process_with_retry(_process)
    if result is None:
        raise gr.Error("处理失败。请检查输入参数并重试。如果问题持续存在，请联系支持团队。")
    return result

def process_dc(vton_img, garm_img, category, n_samples, n_steps, image_scale, seed):
    print(f"发送到 API 的参数: vton_img={vton_img}, garm_img={garm_img}, category={category}, n_samples={n_samples}, n_steps={n_steps}, image_scale={image_scale}, seed={seed}")
    if category == '上半身':
        category = 0
    elif category == '下半身':
        category = 1
    else:
        category =2
    def _process():
        result = client.predict(
            vton_img=handle_file(vton_img),
            garm_img=handle_file(garm_img),
            n_samples=n_samples,
            n_steps=n_steps,
            category=category_dict[category],
            image_scale=image_scale,
            seed=seed,
            api_name="/process_dc"
        )
        print("API返回结果 (DC):", result)
        return process_images(result) if result else None

    result = process_with_retry(_process)
    if result is None:
        raise gr.Error("处理失败。请检查输入参数并重试。如果问题持续存在，请联系支持团队。")
    return result

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("# OOTDiffusion演示")
    with gr.Row():
        gr.Markdown("## 半身")
    with gr.Row():
        gr.Markdown("***支持上半身服装***")
    with gr.Row():
        with gr.Column():
            vton_img = gr.Image(label="模特", sources='upload', type="filepath", height=384, value=model_hd)
            example = gr.Examples(
                inputs=vton_img,
                examples_per_page=14,
                examples=[os.path.join(example_path, f'model/model_{i}.png') for i in range(1, 8)] +
                         [os.path.join(example_path, f'model/{i:05d}_00.jpg') for i in [1008, 7966, 5997, 2849, 14627, 9597, 1861]]
            )
        with gr.Column():
            garm_img = gr.Image(label="服装", sources='upload', type="filepath", height=384, value=garment_hd)
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=14,
                examples=[os.path.join(example_path, f'garment/{i:05d}_00.jpg') for i in [3244, 126, 3032, 6123, 2305, 55, 470, 2015, 10297, 7382, 7764, 151, 12562, 4825]]
            )
        with gr.Column():
            result_gallery = gr.Gallery(label='输出', show_label=False, elem_id="gallery", preview=True, scale=1)   
    with gr.Column():
        run_button = gr.Button(value="运行")
        n_samples = gr.Slider(label="图片数量", minimum=1, maximum=4, value=1, step=1)
        n_steps = gr.Slider(label="步骤", minimum=20, maximum=40, value=20, step=1)
        image_scale = gr.Slider(label="引导比例", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed = gr.Slider(label="种子", minimum=-1, maximum=2147483647, step=1, value=-1)
        
    ips = [vton_img, garm_img, n_samples, n_steps, image_scale, seed]
    run_button.click(fn=process_hd, inputs=ips, outputs=[result_gallery])

    with gr.Row():
        gr.Markdown("## 全身")
    with gr.Row():
        gr.Markdown("***支持上半身/下半身/连衣裙; 服装类别必须配对!!!***")
    with gr.Row():
        with gr.Column():
            vton_img_dc = gr.Image(label="模特", sources='upload', type="filepath", height=384, value=model_dc)
            example = gr.Examples(
                label="示例 (上半身/下半身)",
                inputs=vton_img_dc,
                examples_per_page=7,
                examples=[os.path.join(example_path, 'model/model_8.png')] +
                         [os.path.join(example_path, f'model/{i:06d}_0.jpg') for i in [49447, 49713, 51482, 51918, 51962, 49205]]
            )
            example = gr.Examples(
                label="示例 (连衣裙)",
                inputs=vton_img_dc,
                examples_per_page=7,
                examples=[os.path.join(example_path, 'model/model_9.png')] +
                         [os.path.join(example_path, f'model/{i:06d}_0.jpg') for i in [52767, 52472, 53514, 53228, 52964, 53700]]
            )
        with gr.Column():
            garm_img_dc = gr.Image(label="服装", sources='upload', type="filepath", height=384, value=garment_dc)
            category_dc = gr.Dropdown(label="服装类别 (重要选项!!!)", choices=["上半身", "下半身", "连衣裙"], value="上半身")
            example = gr.Examples(
                label="示例 (上半身)",
                inputs=garm_img_dc,
                examples_per_page=7,
                examples=[os.path.join(example_path, f'garment/{i:06d}_1.jpg') for i in [48554, 49920, 49965, 49949, 50181, 49805, 50105]]
            )
            example = gr.Examples(
                label="示例 (下半身)",
                inputs=garm_img_dc,
                examples_per_page=7,
                examples=[os.path.join(example_path, f'garment/{i:06d}_1.jpg') for i in [51827, 51946, 51473, 51515, 51517, 51988, 51412]]
            )
            example = gr.Examples(
                label="示例 (连衣裙)",
                inputs=garm_img_dc,
                examples_per_page=7,
                examples=[os.path.join(example_path, f'garment/{i:06d}_1.jpg') for i in [53290, 53744, 53742, 53786, 53790, 53319, 52234]]
            )
        with gr.Column():
            result_gallery_dc = gr.Gallery(label='输出', show_label=False, elem_id="gallery", preview=True, scale=1)   
    with gr.Column():
        run_button_dc = gr.Button(value="运行")
        n_samples_dc = gr.Slider(label="图片数量", minimum=1, maximum=4, value=1, step=1)
        n_steps_dc = gr.Slider(label="步骤", minimum=20, maximum=40, value=20, step=1)
        image_scale_dc = gr.Slider(label="引导比例", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed_dc = gr.Slider(label="种子", minimum=-1, maximum=2147483647, step=1, value=-1)
        
    ips_dc = [vton_img_dc, garm_img_dc, category_dc, n_samples_dc, n_steps_dc, image_scale_dc, seed_dc]
    run_button_dc.click(fn=process_dc, inputs=ips_dc, outputs=[result_gallery_dc])

block.launch(show_error=True)