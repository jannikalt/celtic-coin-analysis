from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import pandas as pd
from PIL import Image


def _try_load_image(path: str) -> Image.Image | None:
    """Try to load an image from a path, return None if it fails."""
    try:
        if not Path(path).exists():
            return None
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def render_dataset_viewer(dataset_path: str):
    """Render the dataset viewer interface."""
    st.subheader("Dataset Viewer")
    
    if not dataset_path or not dataset_path.strip():
        st.info("Please specify a dataset path in the sidebar to view the dataset.")
        return
    
    dataset_path = dataset_path.strip()
    
    if not Path(dataset_path).exists():
        st.error(f"Dataset file not found: {dataset_path}")
        return
    
    try:
        # Load the dataset
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path, encoding='utf-8')
        elif dataset_path.endswith('.tsv'):
            df = pd.read_csv(dataset_path, sep='\t', encoding='utf-8')
        else:
            st.error("Dataset must be a CSV or TSV file.")
            return
        
        # Verify required columns
        required_cols = ['id', 'label', 'obverse_path', 'reverse_path']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Dataset is missing required columns: {', '.join(missing_cols)}")
            st.info(f"Available columns: {', '.join(df.columns)}")
            return
        
        # Initialize session state for navigation
        if 'selected_label' not in st.session_state:
            st.session_state.selected_label = None
        if 'selected_coin_id' not in st.session_state:
            st.session_state.selected_coin_id = None
        
        # Count coins per label
        label_counts = df['label'].value_counts().sort_index()
        
        st.markdown(f"**Total coins in dataset:** {len(df)}")
        st.markdown(f"**Total labels:** {len(label_counts)}")
        
        # Add search by ID
        st.divider()
        st.markdown("### Search by ID")
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_id = st.text_input("Enter Coin ID", key="search_coin_id", placeholder="e.g., coin_001")
        with search_col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("Search", type="primary"):
                if search_id and search_id.strip():
                    matching_coins = df[df['id'].astype(str).str.contains(search_id.strip(), case=False, na=False)]
                    if len(matching_coins) > 0:
                        if len(matching_coins) == 1:
                            # Go directly to coin details
                            st.session_state.selected_coin_id = matching_coins.iloc[0]['id']
                            st.session_state.selected_label = matching_coins.iloc[0]['label']
                            st.rerun()
                        else:
                            # Show matching coins
                            st.session_state.search_results = matching_coins
                            st.rerun()
                    else:
                        st.warning(f"No coins found matching '{search_id}'")
        
        # Show search results if available
        if 'search_results' in st.session_state and st.session_state.search_results is not None:
            search_df = st.session_state.search_results
            st.divider()
            st.markdown(f"### Search Results ({len(search_df)} coins found)")
            
            if st.button("← Clear Search", key="clear_search"):
                st.session_state.search_results = None
                st.rerun()
            
            # Display found coins in a grid
            cols_per_row = 4
            coins = search_df.to_dict('records')
            
            for i in range(0, len(coins), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(coins):
                        coin = coins[i + j]
                        with col:
                            # Load images
                            obv_img = _try_load_image(str(coin['obverse_path']))
                            rev_img = _try_load_image(str(coin['reverse_path']))
                            
                            # Display images
                            with st.container():
                                img_cols = st.columns(2)
                                with img_cols[0]:
                                    if obv_img is not None:
                                        st.image(obv_img, caption="Obv", width="stretch")
                                    else:
                                        st.caption("! Obv")
                                with img_cols[1]:
                                    if rev_img is not None:
                                        st.image(rev_img, caption="Rev", width="stretch")
                                    else:
                                        st.caption("! Rev")
                                
                                st.caption(f"**{coin['id']}**")
                                st.caption(f"Label: {coin['label']}")
                                
                                if st.button(f"View", key=f"search_coin_{coin['id']}", width="stretch"):
                                    st.session_state.selected_coin_id = coin['id']
                                    st.session_state.selected_label = coin['label']
                                    st.session_state.search_results = None
                                    st.rerun()
            return
        
        st.divider()
        
        # Show label overview if no label is selected
        if st.session_state.selected_label is None:
            st.markdown("### Labels Overview")
            st.markdown("Click on a label to view its coins.")
            
            # Display labels in a grid
            cols_per_row = 4
            labels = sorted(label_counts.index)
            
            for i in range(0, len(labels), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(labels):
                        label = labels[i + j]
                        count = label_counts[label]
                        with col:
                            if st.button(f"{label}\n({count} coins)", key=f"label_{label}", width="stretch"):
                                st.session_state.selected_label = label
                                st.session_state.selected_coin_id = None
                                st.rerun()
        
        # Show coins for selected label
        elif st.session_state.selected_coin_id is None:
            label = st.session_state.selected_label
            label_df = df[df['label'] == label]
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("← Back to Labels", width="content"):
                    st.session_state.selected_label = None
                    st.rerun()
            with col2:
                st.markdown(f"### Label: **{label}** ({len(label_df)} coins)")
            
            st.divider()
            st.markdown("Click on a coin to view its metadata.")
            
            # Display coins in a grid
            cols_per_row = 4
            coins = label_df.to_dict('records')
            
            for i in range(0, len(coins), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(coins):
                        coin = coins[i + j]
                        with col:
                            # Load images
                            obv_img = _try_load_image(str(coin['obverse_path']))
                            rev_img = _try_load_image(str(coin['reverse_path']))
                            
                            # Create a container for the coin
                            with st.container():
                                # Display images side by side
                                img_cols = st.columns(2)
                                with img_cols[0]:
                                    if obv_img is not None:
                                        st.image(obv_img, caption="Obverse", width="content")
                                    else:
                                        st.caption("⚠️ Obverse not found")
                                with img_cols[1]:
                                    if rev_img is not None:
                                        st.image(rev_img, caption="Reverse", width="content")
                                    else:
                                        st.caption("⚠️ Reverse not found")
                                
                                # Button to view details
                                if st.button(f"View Details", key=f"coin_{coin['id']}", width="content"):
                                    st.session_state.selected_coin_id = coin['id']
                                    st.rerun()
        
        # Show detailed view for selected coin
        else:
            coin_id = st.session_state.selected_coin_id
            coin_df = df[df['id'] == coin_id]
            
            if len(coin_df) == 0:
                st.error(f"Coin with ID {coin_id} not found.")
                st.session_state.selected_coin_id = None
                st.rerun()
                return
            
            coin = coin_df.iloc[0].to_dict()
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("← Back to Coins", width="content"):
                    st.session_state.selected_coin_id = None
                    st.rerun()
            with col2:
                st.markdown(f"### Coin Details: **{coin['id']}**")
            
            st.divider()
            
            # Display images
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Obverse**")
                obv_img = _try_load_image(str(coin['obverse_path']))
                if obv_img is not None:
                    st.image(obv_img, width="content")
                else:
                    st.warning("Image not found")
                    st.caption(f"Path: {coin['obverse_path']}")
            
            with col_right:
                st.markdown("**Reverse**")
                rev_img = _try_load_image(str(coin['reverse_path']))
                if rev_img is not None:
                    st.image(rev_img, width="content")
                else:
                    st.warning("Image not found")
                    st.caption(f"Path: {coin['reverse_path']}")
            
            st.divider()
            
            # Display all metadata
            st.markdown("### Metadata")
            
            # Create a clean display of all columns
            metadata_df = pd.DataFrame([coin]).T
            metadata_df.columns = ['Value']
            metadata_df.index.name = 'Field'
            
            st.dataframe(metadata_df, width="content")
            
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        import traceback
        st.code(traceback.format_exc())
