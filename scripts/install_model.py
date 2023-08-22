import urllib.request
import shutil
...


import fire

def __main__():
    model_url = ...


    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

if __name__ == "__main__":
    fire.Fire(__main__)