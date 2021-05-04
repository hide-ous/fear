from psaw import PushshiftAPI


if __name__ == '__main__':

    api = PushshiftAPI()
    gen = api.search_comments(q='OP', subreddit='askreddit')

    max_response_cache = 1000
    cache = []

    for c in gen:
        cache.append(c)

        # Omit this test to actually return all results. Wouldn't recommend it though: could take a while, but you do you.
        if len(cache) >= max_response_cache:
            break

    # If you really want to: pick up where we left off to get the rest of the results.
    if False:
        for c in gen:
            cache.append(c)