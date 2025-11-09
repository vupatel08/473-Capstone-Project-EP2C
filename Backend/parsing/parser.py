# Basic pdf parsing script using MinerU.

import sys
import os
import copy
import json
from pathlib import Path

from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make


# Parsing from Opendatalab:
# Copyright (c) Opendatalab. All rights reserved.
def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'en' (English)
    model: str = "pipeline" # Backend model for parsing. "pipeline" or "vlm", default "pipeline"
):
    
    assert model in ("pipeline", "vlm"), "Invalid model type. Use 'pipeline' or 'vlm'."

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id=0, end_page_id=None)
        pdf_bytes_list[idx] = new_pdf_bytes

    if model == "vlm":
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = [], [], [], [], []
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            lang = p_lang_list[idx] if idx < len(p_lang_list) else "en"
            result, img_lst, pdfs, langs, ocr = \
                vlm_doc_analyze(pdf_bytes, lang, parse_method="auto", formula_enable=True, table_enable=True)
            infer_results.append(result)
            all_image_lists.append(img_lst)
            all_pdf_docs.append(pdfs)
            lang_list.append(langs)
            ocr_enabled_list.append(ocr)
    else:
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = \
            pipeline_doc_analyze(pdf_bytes_list, p_lang_list, parse_method="auto", formula_enable=True, table_enable=True)

    for idx, model_list in enumerate(infer_results):
        model_json = copy.deepcopy(model_list)
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, "auto")
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_doc = all_pdf_docs[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]

        middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True)

        pdf_info = middle_json["pdf_info"]

        pdf_bytes = pdf_bytes_list[idx]
        _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox=True, f_draw_span_bbox=True, f_dump_orig_pdf=True,
            f_dump_md=True, f_dump_content_list=True, f_dump_middle_json=True, f_dump_model_output=True,
            f_make_md_mode=MakeMode.MM_MD, middle_json=middle_json, model_output=model_json, is_pipeline=True
        )

    return

# Copyright (c) Opendatalab. All rights reserved.
def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        is_pipeline=True
):
    
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )

    logger.info(f"local output dir is {local_md_dir}")

    return

# Copyright (c) Opendatalab. All rights reserved.
def parse_doc(
        path_list: list[Path],
        output_dir="parse_output",
        lang="en",
        model="pipeline"
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'en', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            model=model
        )
    except Exception as e:
        logger.exception(e)

    return


if __name__ == "__main__":
    # Enter hard-coded pdf/output paths here...
    pdf_paths = []
    output_path = None

    # Ensure all paths are Path objects.
    for i in range(len(pdf_paths)):
        if not isinstance(pdf_paths[i], Path):
            if not isinstance(pdf_paths[i], str):
                print("pdf paths must be Path objects or strings.", file=sys.stderr)
                exit()
            pdf_paths[i] = Path(pdf_paths[i])
    if isinstance(output_path, str):
        output_path = Path(output_path)
    elif output_path and (not isinstance(output_path, Path)):
        print("output directory path must be a Path object or string.", file=sys.stderr)
        exit()

    if not (pdf_paths and output_path):
        if len(sys.argv) == 1:
            print("Usage: python <script>.py \"<pdf path 1>\" \"<pdf path 2>\" ... \"<output dir path>\"")
            exit()

        for path in sys.argv[1:-1]:
            pdf_paths.append(Path(path))

        output_path = Path(sys.argv[-1])

    # Parse the given pdfs.
    parse_doc(pdf_paths, output_path, lang="en") # model defaults to "pipeline"
