import streamlit as st
import core

def run_app():
    st.title("URL Redirect Matchmaker")

    st.sidebar.header("Upload Files")
    origin_file = st.sidebar.file_uploader("Choose a file for origin URLs", type=['csv'])
    destination_file = st.sidebar.file_uploader("Choose a file for destination URLs", type=['csv'])

    if origin_file and destination_file:
        origin_data = core.load_csv(origin_file)
        destination_data = core.load_csv(destination_file)

        preprocessed_origin = core.preprocess_data(origin_data)
        preprocessed_destination = core.preprocess_data(destination_data)

        origin_embeddings = core.compute_embeddings(preprocessed_origin['content'])
        destination_embeddings = core.compute_embeddings(preprocessed_destination['content'])

        destination_index = core.create_faiss_index(destination_embeddings)
        matches, distances = core.find_matches(origin_embeddings, destination_index)

        st.write("Matched URLs:")
        for i, (match, distance) in enumerate(zip(matches[0], distances[0])):
            st.write(f"Origin URL: {preprocessed_origin['url'][i]} -> Destination URL: {preprocessed_destination['url'][match]} with distance: {distance}")

if __name__ == "__main__":
    run_app()
