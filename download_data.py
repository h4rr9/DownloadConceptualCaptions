"""Download Conceptual Captions Efficiently."""
import pandas as pd
import click
from PIL import Image
import requests
import zlib
import os
import io
import shelve
import magic
from multiprocessing import Pool
from tqdm import tqdm

headers = {
    "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
    "X-Forwarded-For": "64.18.15.200",
}


def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)


def df_multiprocess(df, processes, chunk_size, func, dataset_name):
    """Download process."""
    print("Generating parts...")
    with shelve.open(
        "%s_%s_%s_results.tmp" % (dataset_name, func.__name__, chunk_size)
    ) as results:
        pbar = tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = (
            (index, df[i : i + chunk_size], func)
            for index, i in enumerate(range(0, len(df), chunk_size))
            if index not in finished_chunks
        )
        print(
            int(len(df) / chunk_size),
            "parts.",
            chunk_size,
            "per part.",
            "Using",
            processes,
            "processes",
        )

        pbar.desc = "Downloading"
        with Pool(processes) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_df_split_apply, pool_data, 2)
            ):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print("Finished Downloading.")
    return


def _file_name(row):
    """Create unique name based on url."""
    return "%s/%s_%s" % (
        row["folder"],
        row.name,
        (zlib.crc32(row["url"].encode("utf-8")) & 0xFFFFFFFF),
    )


def check_mimetype(row):
    """For checking mimetypes separately without download."""
    if os.path.isfile(str(row["file"])):
        row["mimetype"] = magic.from_file(row["file"], mime=True)
        row["size"] = os.stat(row["file"]).st_size
    return row


def check_download(row):
    """Don't download image, just check with a HEAD request, can't resume.

    Can use this instead of download_image to get HTTP status codes.
    """
    fname = _file_name(row)
    try:
        # not all sites will support HEAD
        response = requests.head(
            row["url"],
            stream=False,
            timeout=5,
            allow_redirects=True,
            headers=headers,
        )
        row["status"] = response.status_code
        row["headers"] = dict(response.headers)
    except Exception:
        # log errors later, set error as 408 timeout
        row["status"] = 408
        return row
    if response.ok:
        row["file"] = fname
    return row


def download_image(row):
    """Download image."""
    fname = _file_name(row)
    # Skip Already downloaded, retry others later
    if os.path.isfile(fname):
        row["status"] = 200
        row["file"] = fname
        row["mimetype"] = magic.from_file(row["file"], mime=True)
        row["size"] = os.stat(row["file"]).st_size
        return row

    try:
        # use smaller timeout to skip errors, but can result in failed download
        response = requests.get(
            row["url"],
            stream=False,
            timeout=10,
            allow_redirects=True,
            headers=headers,
        )
        row["status"] = response.status_code
        # row['headers'] = dict(response.headers)
    except Exception:
        # log errors later, set error as 408 timeout
        row["status"] = 408
        return row

    if response.ok:
        try:
            with open(fname, "wb") as out_file:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True
                image = Image.open(io.BytesIO(response.content))
                image = image.resize((32, 32), Image.LANCZOS)
                out_file.write(image.tobytes())
            row["mimetype"] = magic.from_file(fname, mime=True)
            row["size"] = os.stat(fname).st_size
        except Exception:
            # This is if it times out during a download or decode
            row["status"] = 408
            return row
        row["file"] = fname
    return row


def open_tsv(fname, folder):
    """Open tsv file."""
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep="\t", names=["caption", "url"], usecols=range(1, 2))
    df["folder"] = folder
    print("Processing", len(df), " Images:")
    return df


def df_from_shelve(chunk_size, func, dataset_name):
    """Generate Dataset from results."""
    print("Generating Dataframe from results...")
    with shelve.open(
        "%s_%s_%s_results.tmp" % (dataset_name, func.__name__, chunk_size)
    ) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    return df


@click.command()
@click.option("--num_processes", default=12, help="the number of processes")
@click.option(
    "--images_per_part",
    default=100,
    help="The number of images per chunk. Changing this value will reset progress",
)
@click.argument("data_split_tsv")
@click.argument("split_name")
def main(num_processes, images_per_part, data_split_tsv, split_name):
    data_name = split_name
    df = open_tsv(data_split_tsv, data_name)
    df_multiprocess(
        df=df,
        processes=num_processes,
        chunk_size=images_per_part,
        func=download_image,
        dataset_name=data_name,
    )
    df = df_from_shelve(
        chunk_size=images_per_part, func=download_image, dataset_name=data_name
    )
    df.to_csv(
        "downloaded_%s_report.tsv.gz" % data_name,
        compression="gzip",
        sep="\t",
        header=False,
        index=False,
    )
    print("Saved.")


if __name__ == "__main__":
    main()
