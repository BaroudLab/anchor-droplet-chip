import asyncio
import os
import shutil

import fire
import urllib3
import zenodopy


async def download_url_to_file(
    url,
    file_path,
):
    if os.path.exists(file_path):
        print(f"File exists: {file_path}")
        return file_path
    print(f"Downloading {file_path}")
    c = urllib3.PoolManager()
    with c.request(
        "GET",
        url,
        preload_content=False,
    ) as resp, open(file_path, "wb") as out_file:
        shutil.copyfileobj(resp, out_file)
    resp.release_conn()
    print(f"Saved {file_path}")
    return file_path


async def main(projectid: str = None, destination: str = "."):
    if not os.path.exists(destination):
        os.mkdir(destination)
    zeno = zenodopy.Client()
    _ = zeno.list_projects
    zeno.set_project(projectid)
    files = zeno.list_files
    tasks = [
        download_url_to_file(
            f["links"]["download"] + f"?access_token={zeno._get_key()}",
            os.path.join(destination, f["filename"]),
        )
        for f in files["files"]
    ]
    files = asyncio.gather(*tasks)


if __name__ == "__main__":
    fire.Fire(main)
