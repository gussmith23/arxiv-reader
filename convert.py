# Copyright 2022 Tristan Behrens.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import os
import glob
from typing import List, Optional, Tuple
import bs4
from bs4 import BeautifulSoup
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import scipy
import numpy as np
import sys


def main(arxiv_id: Optional[str] = None, remove_ranges: List[Tuple[int,int]] = []):
    # For now, expect that arxiv_id is set. In the future, would be nice to support other ways to get papers.
    assert arxiv_id is not None, "arxiv_id must be set."
    paper_id = arxiv_id

    # Make sure there is a temp directory. Delete it if it exists. Switch into it.
    # TODO(@gussmith23): Use tempfile.
    if os.path.exists("temp"):
        os.system("rm -rf temp")
    os.mkdir("temp")
    os.chdir("temp")

    # Download the paper as .tar.gz
    print(f"Downloading paper {paper_id}...")
    os.system(f"arxiv-downloader --id {paper_id} --source")

    # Find the .tar.gz file.
    try:
        tar_gz_file = glob.glob(f"{paper_id}*.tar.gz")[0]
    except:
        print(
            f"Could not find the .tar.gz file for {paper_id}. Maybe the download did not work?"
        )
        exit

    # Extract the .tar.gz file to a temp folder.
    os.system(f"tar -xzf {tar_gz_file}")

    # Convert to HTML.
    get_sentences_from_tex(paper_id=paper_id, remove_ranges=remove_ranges)

    # Go back. Up one level.
    os.chdir("..")

    # Convert to wav.
    convert_sentences_to_wav(paper_id=paper_id)

    # Remove temp folder.
    os.system(f"rm -rf temp")


def get_sentences_from_tex(paper_id: str, remove_ranges: List[Tuple[int,int]] = []):
    # Find all the .tex files in the temp folder.
    tex_files = glob.glob(f"*.tex")

    # Find all the tex files whose content start with the string \documentclass.
    documentclass_files = []
    for tex_file in tex_files:
        with open(tex_file, "r") as f:
            if f.readline().startswith("\documentclass"):
                documentclass_files.append(tex_file)
    assert len(documentclass_files) == 1, "There should be only one documentclass file."
    documentclass_file = documentclass_files[0]

    # Sort them and reverse them, so that, as we remove lines, we don't change
    # the lower line numbers. This assumes non-overlapping ranges.
    remove_ranges = reversed(sorted(remove_ranges, key=lambda x: x[0]))

    for (start, end) in remove_ranges:
        with open(documentclass_file, "r") as f:
            lines = f.readlines()
        lines = lines[:start] + lines[end:]
        with open(documentclass_file, "w") as f:
            f.writelines(lines)

    # Convert the .tex file to .md file.
    os.system(f"pandoc {documentclass_file} -o {paper_id}.html -t html5")

    # Load the .html file with BeautifulSoup4.
    with open(f"{paper_id}.html", "r") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")

    # Cleanup. Rigorouslly, we want to remove a lot of tags.
    for element in soup.find_all("span", class_="citation"):
        element.decompose()
    for element in soup.find_all("span", class_="math inline"):
        element.decompose()
    for element in soup.find_all("span", class_="math display"):
        element.decompose()
    for element in soup.find_all("div", class_="figure"):
        element.decompose()
    for element in soup.find_all("div", class_="figure*"):
        element.decompose()
    for element in soup.find_all("div", class_="thebibliography"):
        element.decompose()
    for element in soup.find_all("div", class_="center"):
        element.decompose()
    for element in soup.find_all("section", class_="footnotes"):
        element.decompose()
    for element in soup.find_all("a"):
        element.decompose()
    for element in soup.find_all("figure"):
        element.decompose()
    for element in soup.find_all("table"):
        element.decompose()
    for element in soup.find_all("table*"):
        element.decompose()
    for element in soup.find_all("div", class_="description"):
        element.decompose()
    for element in soup.find_all("tbody"):
        element.decompose()

    # Elements to unwrap (i.e. remove the tag, but keep the contents).
    for tag in ["span", "strong", "em", "i", "b", "li", "code", "ul", "ol"]:
        for element in soup.find_all(tag):
            element.unwrap()

    # Write the .html file back.
    with open(f"{paper_id}_cleaned.html", "w") as f:
        f.write(soup.prettify())

    # Read that .html file back line by line.
    with open(f"{paper_id}_cleaned.html", "r") as f:
        lines = f.readlines()

    # Convert to sentences. Go through the lines
    sentences = []
    accumumlated_sentence = ""
    for line in lines:
        if line.startswith("<"):
            # Opening tags that we expect.
            if (
                line.startswith("<p")
                or line.startswith("<h1")
                or line.startswith("<h2")
                or line.startswith("<h3")
                or line.startswith("<h4")
            ):
                pass

            # Closing tags that we expect.
            elif (
                line.startswith("</p>")
                or line.startswith("</h1>")
                or line.startswith("</h2>")
                or line.startswith("</h3>")
                or line.startswith("</h4>")
            ):
                accumumlated_sentence = accumumlated_sentence.replace("\n", " ")

                # Add spaces around hyphens. We could also delete them entirely.
                # They're pronounced very strangely by the model.
                accumumlated_sentence = accumumlated_sentence.replace("-", " - ")

                # Split by period so that we can insert a pause.
                for x in accumumlated_sentence.split("."):
                    sentences.append(x.strip())
                    sentences.append("<PAUSE>")

                # Start over and add pause.
                accumumlated_sentence = ""
                sentences.append("<PAUSE>")

            else:
                print(f"Unexpected HTML tag: {line}")

        # Accumulate texts.
        else:
            accumumlated_sentence += line

    # Write to a file.
    with open(f"{paper_id}_sentences.txt", "w") as f:
        for sentence in sentences:
            if sentence.strip() != "":
                f.write(sentence + "\n")

    # Done.
    return sentences


def convert_sentences_to_wav(paper_id: str):
    # Load lines from the .txt file.
    with open(f"temp/{paper_id}_sentences.txt", "r") as f:
        sentences = f.readlines()

    # Load the model.
    print("Loading TTS model...")
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/fastspeech2-en-ljspeech",
        arg_overrides={"vocoder": "hifigan", "fp16": False},
    )
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(models, cfg)

    # Generate line by line.
    print("Generating...")
    full_wave_file = []
    rate = 44100
    for text in sentences:
        text = text.strip()

        print(f'Text: "{text}"')
        if text == "":
            continue

        # Insert a pause.
        if text == "<PAUSE>":
            full_wave_file.extend(np.zeros(rate))
            continue

        # Create the sample.
        sample = TTSHubInterface.get_model_input(task, text)
        wav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)

        # Map wav from torch tensor to numpy array.
        wav = wav.numpy()

        # Append.
        full_wave_file.extend(wav)

    # Convert to numpy.
    full_wave_file = np.array(full_wave_file, dtype=np.float32)

    # Save the generated audio to a file.
    wav_path = f"{paper_id}.wav"
    print(f"Saving {wav_path}")
    scipy.io.wavfile.write(wav_path, rate, full_wave_file)
    print("Done.")

    print("Converting to mp3...")
    os.system(f"ffmpeg -y -i {wav_path} {paper_id}.mp3")
    print("Done.")


# Call the main method.
if __name__ == "__main__":
    import argparse
    import ast

    parser = argparse.ArgumentParser(description="Convert a paper to audio.")
    parser.add_argument(
        "--arxiv_id",
        type=str,
        # TODO(@gussmith23): In the future, it would be nice to support papers
        # from other sources.
        required=True,
        help="The arxiv ID of the paper to convert.",
    )
    parser.add_argument(
        "--remove_ranges",
        type=str,
        default="[]",
        help=(
            "Remove lines from LaTeX source. This is the easiest way to get"
            " around https://github.com/jgm/pandoc/issues/4746. Expects a list"
            ' of pairs "[(start0,end0), (start1,end1), ...]". Will remove'
            " lines from startn to endn-1 (Python half-open range style) for"
            " each (start,end) pair. Assumes ranges are"
            " non-overlapping; things will break if this is not the case!"
            " Only removes lines from the top-level file."
        ),
    )

    args = parser.parse_args()

    main(arxiv_id=args.arxiv_id, remove_ranges=ast.literal_eval(args.remove_ranges))
