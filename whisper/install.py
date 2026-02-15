from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv()

output_dir=os.environ.get("DIR")
model=OVModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base", export=True, trust_remote_code=True)
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("openai/whisper-base")
export_tokenizer(tokenizer, output_dir)