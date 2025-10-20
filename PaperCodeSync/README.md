# PaperCodeSync

Pipeline to handle the synchronization layer between the uploaded research paper and the generated code repo.
After researching how OverLeaf does their sync between LaTeX and display, this is the pipeline I came up with:

## Steps
1) Parse the Paper
- First we have to take the uploaded research paper and parse it into its sections. This is pretty similar to the main pipeline's step of parsing the paper, so I might be able to just use the result of that step directly. I've looked into a few different parsers, namely, mineru, GROBID, and docling. Mineru takes a long time to parse and in the end it failed for me when I tested it. Docling's result is an .md file which is a bit rudimentary. I think GROBID might be the way to go, just have to figure it out.

2) Parse the Code Repo
- Second, we have to take our generated code repoistory and parse each file into symbols. As in functions, classes, spans of code, etc. We have to walk through each file of the codebase, detect which langauge it's in (via its extension), then parse each file using "Tree-sitter" giving us a bag of words for each symbol which will make the next step easier.

3) Rank the Sections for each Symbol
- Third, we will create a searchable text index (using either BM25 or TF-IDF). Each symbol from step 2 forms a query using its bag of words. Now for each query, compare it against every paper chunk in the index, retrieve the top 10 paper chunks that share the most important overlapping terms, and finally, save their BM25 or TF-IDF scores (the higher meaning more text overlaps). Then run a second scoring pass to try capturing meaning. To do this, we can encode both the query and the paper chunks into numeric vectors using a sentence-embedding model. Next, we compute the cosine similarity between the symbol and chunk vector. So now we have a score for textual overlap and a score for semantic similarity. Then combine these two scores into one (I can make how much each score weighs ultimatelly as an adjustable knob). Now we can just use the highest score and save the second/third highest as alternatives (maybe?).

4) Generate the Alignment Map 
- Lastly, we want to make a .json map that will be the key into syncing up the paper and its generated code base. This will have to be tweaked a bit as the UI is being made. 