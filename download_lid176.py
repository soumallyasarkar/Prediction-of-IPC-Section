import requests # type: ignore

url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
output_path = "lid.176.bin"

print(f"Downloading {url} ...")
response = requests.get(url, stream=True)
response.raise_for_status()
with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
print(f"Downloaded to {output_path}")
