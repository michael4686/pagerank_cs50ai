import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dist = {page_name : 0 for page_name in corpus}

    # If page has no links, return equal probability for the corpus:
    if len(corpus[page]) == 0:
        for page_name in prob_dist:
            prob_dist[page_name] = 1 / len(corpus)
        return prob_dist

    # Probability of picking any page at random:
    random_prob = (1 - damping_factor) / len(corpus)

    # Probability of picking a link from the page:
    link_prob = damping_factor / len(corpus[page])

    # Add probabilities to the distribution:
    for page_name in prob_dist:
        prob_dist[page_name] += random_prob

        if page_name in corpus[page]:
            prob_dist[page_name] += link_prob

    return prob_dist



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {page: 0 for page in corpus}
    page = random.choice(list(corpus.keys()))
    
    for _ in range(n):
        page_rank[page] += 1
        model = transition_model(corpus, page, damping_factor)
        pages = list(model.keys())
        probabilities = list(model.values())
        page = random.choices(pages, probabilities)[0]
    
    total_samples = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= total_samples
    
    return page_rank



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    total_pages = len(corpus)
    initial_rank = 1 / total_pages
    ranks = {page: initial_rank for page in corpus}

    convergence = False
    while not convergence:
        new_ranks = {}

        for page in corpus:
            new_rank = (1 - damping_factor) / total_pages

            for linking_page, links in corpus.items():
                if len(links) == 0:
                    # Interpret pages with no links as having one link to all pages
                    new_rank += damping_factor * ranks[linking_page] / total_pages

                if page in links:
                    new_rank += damping_factor * ranks[linking_page] / len(links)

            new_ranks[page] = new_rank
            
        # Check for convergence
        convergence = all(abs(new_ranks[page] - ranks[page]) <= 0.001 for page in corpus)

        # Update the current ranks with the new ranks
        ranks = new_ranks


    rank_sum = sum(ranks.values())
    pagerank = {page: rank / rank_sum for page, rank in ranks.items()}
    return pagerank


if __name__ == "__main__":
    main()

