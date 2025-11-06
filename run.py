import os
#os.chdir('/content/index-tts')
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=True, use_cuda_kernel=True, use_deepspeed=False)

output_lines = []

with open("input/*seg.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]  # remove blank lines

i = 0
while i < len(lines):
    if lines[i].isdigit():  # Line number
        number = lines[i]
        text = lines[i + 1]  # First part of text
        # Merge with next line if the following isn't a number
        if i + 2 < len(lines) and not lines[i + 2].isdigit():
            text += " " + lines[i + 2]
            i += 1  # Skip extra text line
        #output_lines.append(f"{number} {text}")
        output_lines.append(f"{text}")
        i += 2
    else:
        i += 1

# Print or write back to file
j = 0
while j < len(output_lines):
	k = j+1
	this_text = output_lines[j]
	tts.infer(spk_audio_prompt='input/Segments/Dad-is-a-hero_S01E01-' + k + '.wav', text=this_text, output_path="../output" + k + ".wav", verbose=True)
#for line in output_lines:
#    print(line)


#text1 = "That's odd. I don't sense the young master nearby. Did he return to his mansion? No, wait."
#text2 = "No, Young Master!"
#text3 = "Mio, calm down!"
#text4 = "B-But, but." 
#text5 = "His presence has returned."
#But. Let's head there. Quickly, without the others noticing. Young Master! How long have you been standing there?! Young Master! Hey! You're alive! You're still alive! Y-You're so close. What are you talking. Tomoe, are we under attack? No. You can't be serious! You've died countless times! Uh-huh. But I'm alive. Your actions caused your consciousness to waiver. You appeared to blend into your surroundings. You extended you conciousness to that distant target and then reconstituted yourself? Yes! Y-Young Master, people can only diffuse their conciousness when they d-die. What? What were you doing looked exactly like committing s-suicide! She's crying! But look, I'm actually alive. So there's no problem, right? There's a huge problem.
#tts.infer(spk_audio_prompt='examples/1.wav', text=text1, output_path="gen1.wav", verbose=True)
#tts.infer(spk_audio_prompt='examples/2.wav', text=text2, output_path="gen2.wav", verbose=True)
#tts.infer(spk_audio_prompt='examples/3.wav', text=text3, output_path="gen3.wav", verbose=True)
#tts.infer(spk_audio_prompt='examples/4.wav', text=text4, output_path="gen4.wav", verbose=True)
#tts.infer(spk_audio_prompt='examples/5.wav', text=text5, output_path="gen5.wav", verbose=True)
