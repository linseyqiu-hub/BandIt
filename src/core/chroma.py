"""
Resolve Chroma collection handles from the client stored on app.state.

Called per-request so handles are never stale. If Chroma restarts and
collections are recreated, the next request gets fresh handles
automatically.
"""


def get_collections(app_state):
    """Returns (essays_col, questions_col) from app_state.chroma_client."""
    client = app_state.chroma_client
    essays_col    = client.get_collection("essays",    embedding_function=None)
    questions_col = client.get_collection("questions", embedding_function=None)
    return essays_col, questions_col