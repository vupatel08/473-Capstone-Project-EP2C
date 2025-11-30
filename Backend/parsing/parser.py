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
def _do_parse(
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
def _parse_doc(
        path_list: list[Path],
        langs: list[str],
        output_dir: str,
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
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
        _do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=langs,
            model=model
        )
    except Exception as e:
        logger.exception(e)

    return


def _typecheck(args, types, single_elem = False):
    """
    Ensure all given arguments match the given types.
    args and types can both be a list or both be a single item.
        types can have a tuple of types for each argument.
    single_elem would be True to treat a list as a single element argument.
    Exit the program if any types do not match their respective arguments.
    """

    error_prefix = "EP2C_TYPECHECK: "
    arg_count = 1
    type_count = 1

    # Ensure each argument has a type to check. Also retrieve the number of arguments/types if they are lists.
    if isinstance(args, list):
        arg_count = len(args)

        if not isinstance(types, list):
            if single_elem:
                arg_count = 1
            else:
                print(f"{error_prefix}received more arguments than meta types", file=sys.stderr)
                exit()
            
        type_count = len(types)

    elif isinstance(types, list):
        print(f"{error_prefix}received more meta types than arguments", file=sys.stderr)
        exit()

    # Ensure there are the same number of arguments as types.
    if arg_count != type_count:
        error_str = error_prefix + "more "
        if arg_count > type_count:
            error_str += "arguments than meta types"
        else:
            error_str += "meta types than arguments"
        print(error_str, file=sys.stderr)
        exit()

    if arg_count == 1:
        args = [args]
        types = [types]
    
    # Cross-reference arguments with types.
    for i in range(arg_count):
        curr_type = types[i]
        if not isinstance(curr_type, tuple):
            curr_type = (curr_type,)
        if type(args[i]) not in curr_type:
            print(f"{error_prefix}mismatched types. Argument {i} expected {curr_type} but was {type(args[i])}.")
            exit()

    return


def codegen_prep(doc_paths: list[Path], output_dir: Path) -> list[dict[str: str, str: list[dict[str: str, str: str | Path]]]]:
    context_lst = []
    for i, doc in enumerate(doc_paths):
        curr_doc = str(doc.stem)
        context_lst.append({
            "document": curr_doc,
            "content": []
        })

        paper_output_directory = output_dir / f"/{curr_doc}/auto"

        with open(paper_output_directory / f"/{curr_doc}_content_list.json") as curr_parse:
            parse_content =json.load(curr_parse)
            content_str = ""
            
            for item in parse_content:
                match item["type"]:
                    case "text" | "code" | "title": # check code 
                        content_str += item["text"]
                    case "title":
                        content_str += item["title"]
                    case "table" | "image" | "equation":
                        if content_str:
                            context_lst[i]["content"].append({
                                "type": "text",
                                "text": content_str
                                })
                        context_lst[i]["content"].append({
                            "type": "image",
                            "path": paper_output_directory / 
                                f"/{item["img_path"]}"
                            })
                        
                        if "table_path" in item:
                            content_str = item["table_path"]
                        elif "image_path" in item:
                            content_str = item["image_path"]
                        elif "text" in item:
                            content_str = item["text"]

                        content_str += '\n'
                    case _:
                        continue
                    
                content_str += '\n'

            if content_str:
                context_lst[i]["content"].append({
                    "type": "text",
                    "content": content_str
                })
    
    return context_lst

def ep2c_parse(docs: list[tuple[str | Path, str]], output_path: str | Path):
    """
    Parse a given list of documents with MinerU and save results in the given output directory.

    Parameters:
        docs: list of documents to parse. each element is a tuple containing the path to the document and that document's language.
              Paths can be strings or Path objects. Language options are strings.
        output_path: directory to save results of parsing to. Must be a string or Path object.

    This function will exit the program when incorrect types are given or a language option is unsupported.
    """

    LANGUAGE_OPTIONS = ["ch", "ch_lite", "en", "korean", "japan", "chinese_cht", "ta", "te", "ka",]

    error_prefix = "EP2C_PARSE: "
    if not isinstance(docs, list):
        print(f"{error_prefix}documents must be in a list.", file=sys.stderr) 
        exit()

    path_types = [str, Path]

    if isinstance(output_path, Path):
        output_path = str(output_path)
    elif not isinstance(output_path, str):
        print(f"{error_prefix}output path must be a string or Path.", file=sys.stderr)
        exit()

    doc_paths = []
    langs = []

    # Check arguments.
    for doc_lang in docs:
        # Typecheck all arguments.
        if (not isinstance(doc_lang, tuple)) or \
           (len(doc_lang) != 2) \
           (not ((type(doc_lang[0]) in path_types) and isinstance(doc_lang[1], str))):
            
            print(f"{error_prefix}document list should be a list of (str | Path, str) tuples.", file=sys.stderr)
            exit()
        
        # Ensure languages given are supported.
        if doc_lang[1] not in LANGUAGE_OPTIONS:
            print(f"{error_prefix}\'{doc_lang[1]}\' language option not supported", file=sys.stderr)
            exit()

        doc_paths.append(Path(doc_lang[0]))
        langs.append(doc_lang[1])

    # Parse the documents with MinerU.
    _parse_doc(path_list=doc_paths, langs=langs, output_dir=output_path)

    return
