# Python unzip module with 한글

## 작성 개요

Open dataset를 다운받아 사용할 때 여러개의 zip 파일이 존재할 경우 압축 해제하는게 번거롭다.

데스크탑과 같이 화면이 출력되는 경우에는 불편함이 없지만, 서버에서 작업할 경우 일일이 unzip을 해야하는 것 같아 그냥 python module을 작성하여 사용했다.

## unzip module

```python
def unzip(src, dst_path):
    with zipfile.ZipFile(src, "r") as zf:
        zipInfo = zf.infolist()
        for i in zipInfo:
            try:
                i.filename = i.filename.encode("cp437").decode("euc-kr", "ignore")
                zf.extract(i, dst_path)
            except:
                print(src)
                raise Exception("what?!")
```

python에서 작성한 unzip module이다.

6번째 line에서 한글 파일 이름이나, 폴더의 경우 unzip후 파일 이름이 깨져서 추출될 수 있다. 이를 방지하기 위해 encode, decode를 진행한다.

## 전체 코드

```python
import argparse
import os
import zipfile
from glob import glob

from tqdm import tqdm

def argparses():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", type=str, default=None, required=True, help="insert zip file directory full path"
    )
    parser.add_argument("--extra", type=str, default=None, required=True, help="insert extract root path")

    args = parser.parse_args()

    return args

def unzip(src, dst_path):
    with zipfile.ZipFile(src, "r") as zf:
        zipInfo = zf.infolist()
        for i in zipInfo:
            try:
                i.filename = i.filename.encode("cp437").decode("euc-kr", "ignore")
                zf.extract(i, dst_path)
            except:
                print(src)
                raise Exception("what?!")

def main():
    args = argparses()
    zip_files = glob(os.path.join(args.path, "*.zip"))
    for i in tqdm(zip_files):
        extract_to = os.path.join(args.extra, i.split(sep="/")[-1].split(sep=".zip")[0])
        if not os.path.isdir(extract_to):
            os.mkdir(extract_to)
        unzip(i, extract_to)

if __name__ == "__main__":
    main()
```

사용 방법은 다음과 같다.

- Dependency
    - tqdm
    - glob
- RUN
    - python unzip.py —path zip_source_path —extra dest_path