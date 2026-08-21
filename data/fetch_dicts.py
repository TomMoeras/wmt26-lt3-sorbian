#!/usr/bin/env python3
"""Fetch the two GPL-licensed hunspell dictionaries (never vendored, see NOTICE): soblex
(Upper Sorbian) and dsb-spell (Lower Sorbian). Each is a packaged browser/office extension
archive (.oxt / .xpi -- both plain zip files), downloaded from its own source page and
verified against the pinned sha256 recorded when it was staged for the working repo's Phase 5
B2 (staged/b2_dicts/MANIFEST.json, staged 2026-07-07), then unpacked so hunspell's
`-d ./<lang>` convention resolves: `<out>/<lang>.dic` + `<out>/<lang>.aff`.

Neither source is a git repository, so there is no commit hash to pin against -- both are
pinned by exact download URL + release version + artifact sha256 instead, which is what the
working repo's own manifest recorded.
"""
import argparse
import hashlib
import io
import os
import zipfile

import requests

# Pinned per working-repo staged/b2_dicts/MANIFEST.json (Phase 5 B2, staged 2026-07-07).
SOBLEX_URL = ("https://soblex.de/spellchecker_extension_update/"
              "soblex_hsb_w8_3.09.18_07.03.2026_sc_th_hy.oxt")
SOBLEX_VERSION = "3.09.18"
SOBLEX_SHA256 = "1ff18ed056c70aab50e5fe91633c34d227e5ff6877443792f2cd5963904427bf"

DSB_SPELL_URL = ("https://addons.mozilla.org/firefox/downloads/file/4270292/"
                  "dsb_spell-1.5.0.2resigned1.xpi")
DSB_SPELL_VERSION = "1.5.0.2resigned1"
DSB_SPELL_SHA256 = "2247c545dfe05c91d076e78c8e7081a4ed150ae05219f9d891d8b6a024fc6bf3"

# (lang code used by lt3wmt26.generate.Resources.dict_aff/dict_dic, source url, pinned sha256)
DICTS = [
    ("hsb", SOBLEX_URL, SOBLEX_SHA256),
    ("dsb", DSB_SPELL_URL, DSB_SPELL_SHA256),
]


def fetch(url, sha256, timeout=60):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.content
    got = hashlib.sha256(data).hexdigest()
    assert got == sha256, f"{url}: sha256 mismatch (got {got}, want {sha256})"
    return data


def extract_dic_aff(data, lang, out_dir):
    """Both .oxt (LibreOffice) and .xpi (Firefox) extension archives are plain zip files. The
    internal layout is the extension's own, not something this repo pins -- so this walks the
    archive rather than hard-coding an internal path, and fails loudly if the archive doesn't
    contain exactly one spellcheck .dic and one .aff (an ambiguous archive is a fetch bug, not
    something to guess through). Hyphenation dictionaries (`hyph_*.dic`) ship alongside the
    spellcheck dictionary in the soblex .oxt and are excluded -- they are not hunspell
    spellcheck dictionaries and pair with no .aff."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    dics = [n for n in zf.namelist() if n.lower().endswith(".dic")
            and not os.path.basename(n).lower().startswith("hyph")]
    affs = [n for n in zf.namelist() if n.lower().endswith(".aff")]
    assert len(dics) == 1, f"{lang}: expected exactly one .dic in archive, found {dics}"
    assert len(affs) == 1, f"{lang}: expected exactly one .aff in archive, found {affs}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{lang}.dic"), "wb") as f:
        f.write(zf.read(dics[0]))
    with open(os.path.join(out_dir, f"{lang}.aff"), "wb") as f:
        f.write(zf.read(affs[0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Directory to write <lang>.dic/<lang>.aff into.")
    args = ap.parse_args(argv)
    for lang, url, sha in DICTS:
        data = fetch(url, sha)
        extract_dic_aff(data, lang, args.out)
        print(f"[fetch_dicts] {lang}: fetched + verified ({url})", flush=True)


if __name__ == "__main__":
    main()
