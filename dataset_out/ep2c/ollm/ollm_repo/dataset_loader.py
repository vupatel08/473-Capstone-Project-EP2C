## dataset_loader.py

import os
import json
import requests
import csv
from collections import deque, defaultdict
from typing import List, Dict, Tuple
import hashlib

# Import Dataset class as specified in the Data Structures and Interfaces
# For illustration, define a simple Dataset data class here
from dataclasses import dataclass

@dataclass
class Dataset:
    documents: List[str]
    concepts: List[str]
    relations: List[Tuple[str, str, str]]  # (concept1, relation_type, concept2)
    annotations: Dict[int, List[str]]     # document_id -> list of concepts


class DatasetLoader:
    def __init__(self, config: Dict):
        self.config = config
        # Directory to cache datasets
        self.cache_dir = "cached_datasets"
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_wikipedia(self) -> Dataset:
        """
        Load and process Wikipedia dataset:
        - Perform BFS from 'Main topic classifications' category up to depth 3.
        - Retrieve page titles and summaries for concepts.
        - Collect documents annotated with concepts.
        """
        cache_path = os.path.join(self.cache_dir, "wikipedia_dataset.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            documents = data['documents']
            concepts = data['concepts']
            relations = data['relations']
            annotations = {int(k): v for k, v in data['annotations'].items()}
            return Dataset(documents, concepts, relations, annotations)

        # Step 1: Retrieve categories starting from 'Main topic classifications'
        starting_category = "Main topic classifications"
        max_depth = 3
        category_graph, category_to_id, id_to_category = self._build_category_graph_bfs(starting_category, max_depth)

        # Step 2: Gather pages and summaries for each category
        concepts = list(category_to_id.keys())
        concept_id_map = {c: category_to_id[c] for c in concepts}

        # For each category, get page titles and summaries
        category_pages = self._get_category_pages(concept_to_id=category_to_id, max_pages=5000)

        # Create documents: concatenate title and summary
        documents = []
        annotations = defaultdict(list)  # document index -> list of concepts
        for cat_id, pages in category_pages.items():
            for idx, page in enumerate(pages):
                doc_text = self._combine_title_summary(page['title'], page['summary'])
                documents.append(doc_text)
                # Assign concepts based on category
                annotations[len(documents)-1].append(cat_id)  # Using category id as concept; can map back to name if needed

        # Build relations: parent-child among categories
        relations = []
        for parent_id, child_id in category_graph:
            relations.append((parent_id, "is-a", child_id))
        # Remove duplicates
        relations = list(set(relations))
        # Convert concept IDs back to names
        concepts = list(concept_to_id.keys())

        # Save cache
        data_to_cache = {
            'documents': documents,
            'concepts': concepts,
            'relations': relations,
            'annotations': dict(annotations)
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_cache, f)

        return Dataset(documents, concepts, relations, dict(annotations))

    def load_arxiv(self) -> Dataset:
        """
        Load and process arXiv dataset:
        - Filter papers from 2020-2022 with ≥10 citations.
        - Text from title + abstract.
        - Concepts from arXiv taxonomy/keywords.
        - Map documents to concepts.
        """
        cache_path = os.path.join(self.cache_dir, "arxiv_dataset.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            documents = data['documents']
            concepts = data['concepts']
            relations = data['relations']
            annotations = {int(k): v for k, v in data['annotations'].items()}
            return Dataset(documents, concepts, relations, annotations)

        # Step 1: Load dataset from arXiv (assumed preprocessed locally)
        arxiv_metadata_path = self.config.get('arxiv_metadata_path', 'arxiv_metadata.csv')
        # The CSV should contain at least: paper_id, title, abstract, submission_date, citation_count, primary_categories
        documents = []
        concept_set = set()
        doc_annotations = defaultdict(list)
        with open(arxiv_metadata_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Filter criteria
                year = int(row['submission_date'][:4])
                citations = int(row.get('citation_count', 0))
                if 2020 <= year <= 2022 and citations >= 10:
                    text = f"{row['title']} {row['abstract']}"
                    documents.append(text)
                    # Generate concept(s) from primary categories or keywords
                    concepts_for_doc = self._extract_concepts_from_categories(row['primary_categories'])
                    for c in concepts_for_doc:
                        concept_set.add(c)
                    # Save annotations
                    doc_annotations[len(documents)-1] = concepts_for_doc

        concepts = list(concept_set)

        # Build relations: for illustration, assume hierarchical relation 'is-a' among concepts
        relations = []
        # This step can be extended based on actual arXiv taxonomy
        # For simplicity, treat primary categories as concepts linked to broader categories
        # Alternatively, could connect concepts within the same paper
        # For now, relations are a placeholder
        # TODO: If arXiv has a structured taxonomy, populate accordingly

        # Save cache
        data_to_cache = {
            'documents': documents,
            'concepts': concepts,
            'relations': relations,
            'annotations': dict(doc_annotations)
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_cache, f)

        return Dataset(documents, concepts, relations, dict(doc_annotations))

    def _build_category_graph_bfs(self, root_category: str, max_depth: int):
        """
        Perform BFS traversal on category graph starting from root_category.
        Return list of edges (parent, child).
        Since Wikipedia API does not provide graph directly, use MediaWiki API to query category hierarchy.
        """
        base_url = "https://en.wikipedia.org/w/api.php"
        visited = set()
        queue = deque()
        category_graph = []
        category_to_id = {}
        id_to_category = {}
        category_id_counter = 0

        def get_subcategories(category_title):
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category_title}',
                'cmtype': 'subcat',
                'cmlimit': '500'
            }
            response = requests.get(base_url, params=params).json()
            subcats = response.get('query', {}).get('categorymembers', [])
            return [subcat['title'].replace('Category:', '') for subcat in subcats]

        queue.append((root_category, 0))
        category_to_id[root_category] = category_id_counter
        id_to_category[category_id_counter] = root_category
        category_id_counter += 1

        while queue:
            current_cat, depth = queue.popleft()
            if depth >= max_depth:
                continue
            subcategories = get_subcategories(current_cat)
            for subcat in subcategories:
                if subcat not in category_to_id:
                    category_to_id[subcat] = category_id_counter
                    id_to_category[category_id_counter] = subcat
                    category_id_counter += 1
                parent_id = category_to_id[current_cat]
                child_id = category_to_id[subcat]
                category_graph.append((parent_id, child_id))
                queue.append((subcat, depth + 1))
        return category_graph, category_to_id, id_to_category

    def _get_category_pages(self, concept_to_id: Dict[str, int], max_pages: int = 5000):
        """
        Given category IDs, retrieve pages (titles + summaries) belonging to each category.
        Limit to max_pages per category.
        """
        base_url = "https://en.wikipedia.org/w/api.php"
        results = defaultdict(list)
        for category, cat_id in concept_to_id.items():
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': max_pages,
                'cmtype': 'page'
            }
            response = requests.get(base_url, params=params).json()
            pages = response.get('query', {}).get('categorymembers', [])
            for page in pages:
                page_title = page['title']
                summary = self._get_page_summary(page_title)
                results[cat_id].append({'title': page_title, 'summary': summary})
        return results

    def _get_page_summary(self, page_title: str) -> str:
        """
        Retrieve the summary (extracted as text before first section).
        """
        api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/{}".format(page_title.replace(' ', '_'))
        try:
            resp = requests.get(api_url).json()
            summary = resp.get('extract', '')
            return summary
        except:
            return ""

    def _combine_title_summary(self, title: str, summary: str) -> str:
        """
        Concatenate title and summary to form document text.
        """
        return f"Title: {title}\nSummary: {summary}"

    def _extract_concepts_from_categories(self, categories_str: str) -> List[str]:
        """
        Parse categories string, e.g., primary categories or keywords, into concepts.
        Implement heuristic: split by delimiters, clean.
        """
        concepts = []
        if not categories_str:
            return concepts
        # Example heuristic: split by semicolons or spaces
        for cat in categories_str.split(';'):
            cat_clean = cat.strip()
            if cat_clean:
                concepts.append(cat_clean)
        return concepts
