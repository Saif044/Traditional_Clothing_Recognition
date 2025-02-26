# importing gradio
#!export
from fastai.vision.all import load_learner
import gradio as gr

model = load_learner('models/cloth-recognizer-v2.pkl')

#!export
traditional_dress = (
    'Bangladeshi Lungi',
    'Bangladeshi Panjabi',
    'Bavarian Dirndl-Lederhosen',
    'Chinese Qipao-Cheongsam',
    'Indian Saree-Sari',
    'Indonesian Batik',
    'Japanese Kimono',
    'Korean Hanbok',
    'Mexican Huipil',
    'Middle Eastern Thobe-Dishdasha',
    'Mongolian Deel',
    'Native American Regalia',
    'Nigerian Agbada',
    'Polynesian Lavalava',
    'Russian Sarafan',
    'Scottish Kilt',
    'Thai Chut Thai (Traditional Thai dress)'
)

def recognize_image(image):
  pred, idx, probs = model.predict(image)
  return dict(zip(traditional_dress, map(float, probs)))


image = gr.Image(height=520, width=520)
label = gr.Label()
examples = [
    'test_images/sharee.jpg',
    'test_images/kilt.jpg',
    'test_images/lungi.jpg',
    'test_images/regalia.jpg',
    'test_images/kimono.jpg',
    'test_images/panjabi1.jpg',
    'test_images/panjabi2.jpg',
    'test_images/thobe.jpg'
    ]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False,share=True)
