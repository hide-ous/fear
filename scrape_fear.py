from psaw import PushshiftAPI
from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def yield_content(search_func, prefix, **search_params):
    api_request_generator = search_func(**search_params)
    for content in api_request_generator:
        to_return = content.d_
        to_return['name'] = prefix + to_return['id']
        yield to_return


def scrape_pushshift(search_funcs, prefixes, **search_params):
    for search_func, prefix in zip(search_funcs, prefixes):
        for content in tqdm(yield_content(search_func, prefix, **search_params),
                            'scraping with ' + search_func.__name__):
            yield content


def rehydrate_content_pushshift(ids):
    comment_ids = list(filter(lambda x: x.startswith('t1_'), ids))
    submission_ids = list(filter(lambda x: x.startswith('t3_'), ids))
    api = PushshiftAPI()
    # need to chunkize because of:
    # NotImplementedError: When searching by ID, number of IDs must be
    # fewer than the max number of objects in a single request (1000).
    for chunk in chunks(submission_ids, 1000):
        for submission in yield_content(api.search_submissions, 't3_', ids=chunk):
            yield submission
    for chunk in chunks(comment_ids, 1000):
        for comment in yield_content(api.search_comments, 't1_', ids=chunk):
            yield comment


def rehydrate_parents_pushshift(things):
    parent_ids = set()
    for thing in things:
        if 'parent_id' in thing:
            parent_ids.add(thing['parent_id'])
    for parent in tqdm(rehydrate_content_pushshift(parent_ids),
                       "rehydrating parents",
                       len(parent_ids)):
        yield parent



def get_user_pushshift(user):
    api = PushshiftAPI()
    search_funcs = [api.search_submissions, api.search_comments]
    prefixes = ['t3_', 't1_']
    search_params = dict(author=user, )
    yield from scrape_pushshift(
        search_funcs=search_funcs,
        prefixes=prefixes,
        **search_params
    )


if __name__ == '__main__':
    print(get_user_pushshift('hide_ous'))

