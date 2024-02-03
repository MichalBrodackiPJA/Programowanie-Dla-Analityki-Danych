import streamlit as st
from streamlit_option_menu import option_menu

# 1. sidebar menu

# with st.sidebar:
#     selected = option_menu(
#         menu_title='Main Menu', # wymagane
#         options = ['Home', 'Projects', 'Contact'], # wymagane
#         icons= ['house', 'book', 'envelope'],
#         menu_icon = 'cast',
#         default_index=0 # na której stronie będziemy na początku
#     )

#     if selected == 'Home':
#         st.title(f'You have selected {selected}')
#     if selected == 'Projects':
#         st.title(f'You have selected {selected}')
#     if selected == 'Contact':
#         st.title(f'You have selected {selected}')


# 2. horizontal menu

selected = option_menu(
        menu_title=None, # wymagane, None oznacza, że go po prostu nie będzie
        options = ['Home', 'Projects', 'Contact'], # wymagane
        icons= ['house', 'book', 'envelope'],
        menu_icon = 'cast',
        default_index=0, # na której stronie będziemy na początku
        orientation='horizontal'
    )