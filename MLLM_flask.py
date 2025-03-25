from flask import Flask, request, jsonify
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load model and processor (do this once to avoid reloading on each request)
model_dir = "/root/.cache/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct"



# Load model and processor on app start
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir)

@app.route('/describe-image', methods=['POST'])
def describe_image():
    try:
        # Check if 'image' and 'description_prompt' are in the request
        if 'image' not in request.files or 'description_prompt' not in request.form:
            return jsonify({"error": "Invalid input, expected 'image' file and 'description_prompt'"}), 400
        
        # Get the image file and description prompt
        image_file = request.files['image']
        description_prompt = request.form['description_prompt']

        # Read the image into a PIL image
        image = Image.open(io.BytesIO(image_file.read()))

        # Prepare the messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # Pass the in-memory PIL image
                    },
                    {"type": "text", "text": description_prompt},
                ],
            }
        ]

        # Process the input for the model
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to the same device as the model
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate the output
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Return the generated description as a JSON response
        return jsonify({"description": output_text[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8310)
