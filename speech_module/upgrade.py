import os
from dotenv import load_dotenv
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor
# Import the specific tokenizer converter
from openvino_tokenizers import convert_tokenizer 
import openvino as ov

load_dotenv()
output_dir = os.environ.get("DIR", "./whisper-ov-model-v2")
model_id = "sandy1990418/whisper-large-v3-turbo-chinese"

model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, compile=True, device="GPU", export=True)
model.save_pretrained(output_dir)

processor = AutoProcessor.from_pretrained(model_id, feature_size=128)
ov_tokenizer, ov_detokenizer = convert_tokenizer(processor.tokenizer, with_detokenizer=True)

ov.save_model(ov_tokenizer, os.path.join(output_dir, "openvino_tokenizer.xml"))
ov.save_model(ov_detokenizer, os.path.join(output_dir, "openvino_detokenizer.xml"))

print(f"Export complete. Check {output_dir} for openvino_detokenizer.xml")