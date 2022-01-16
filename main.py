import streamlit as st
import pandas as pd
from streamlit.type_util import data_frame_to_bytes
from st_aggrid import AgGrid
from rdkit import Chem
from rdkit.Chem import Draw
from dhclassifier import DHClassifier
import numpy as np
from rdkit.Chem.Draw import IPythonConsole
from PIL import Image

st.set_page_config(page_title="Drug Herbicide Classifier", page_icon ="💠", layout="wide")
DrugLabel      = "💊"
HerbicideLabel = "🌿"
predictions = ["XG_Drug","XG_Herbicide","LR_Drug","LR_Herbicide","RF_Drug","RF_Herbicide","SVM_Drug","SVM_Herbicide"]
dataframe  = pd.DataFrame()
user_input = None
st.title('Drug Herbicide Classifier ') #Set Tittle
st.write(
        """
        ## Explore our Models Online
        
        """)

def paginator(label, items, items_per_page=20, on_sidebar=True):
    """Lets the user paginate a set of items.
    Parameters
    ----------
    label : str
        The label to display over the pagination widget.
    items : Iterator[Any]
        The items to display in the paginator.
    items_per_page: int
        The number of items to display per page.
    on_sidebar: bool
        Whether to display the paginator widget on the sidebar.
        
    Returns
    -------
    Iterator[Tuple[int, Any]]
        An iterator over *only the items on that page*, including
        the item's index.
    Example
    -------
    This shows how to display a few pages of fruit.
    >>> fruit_list = [
    ...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
    ...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
    ...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
    ...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
    ...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
    ... ]
    ...
    ... for i, fruit in paginator("Select a fruit page", fruit_list):
    ...     st.write('%s. **%s**' % (i, fruit))
    """

    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = len(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    page_format_func = lambda i: "Page %s" % i
    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    import itertools
    return itertools.islice(enumerate(items), min_index, max_index)

@st.cache
def fetch_and_clean_data(data):
    return data

#### SELECTION OF INPUT TYPE ####
InputSelector = st.sidebar.selectbox("Select Input Type",("Smiles String","CSV"))                     # Select input Type 
if InputSelector == "Smiles String":
        user_input         = st.sidebar.text_input("Select a Valid Smiles String:", "OC(=O)CNCP(O)(O)=O")             # Siles String INput
        user_input         =  fetch_and_clean_data(user_input)
elif InputSelector == "CSV":
        st.markdown('Select a **.csv** file containing  the SMILES strings of the moleules you wish to analyze.')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
                user_input = pd.read_csv(uploaded_file,header=None)  # Convert csv to panda dataframe
                user_input = user_input.iloc[:, 0]                   # Get all the elements of the first column
                user_input = user_input.tolist()                     # Convert it to list 
         
### Select Visualization tool 
VisualizationSelector = st.sidebar.radio("Show Molecule",('Yes', 'No'))
OutputSelector = st.sidebar.selectbox("Select Outpuy Type",("Simplified","Verbose"))                     # Select input Type 
roundby   = st.sidebar.slider("Decimal Places",min_value=1, max_value=6,value=6)
Threshold = st.sidebar.slider("Threshold",min_value=0.1, max_value=0.9,value=0.5)

##### Work with dataframe ###
if user_input != None:
        dataframe = DHClassifier(user_input)
        if OutputSelector == "Verbose":
                pass
        else:
                dataframe =  dataframe[["SMILES","mol","XG_Drug","XG_Herbicide","LR_Drug","LR_Herbicide","RF_Drug","RF_Herbicide","SVM_Drug","SVM_Herbicide"]]

        #### Remove MOl  ###
        dataframe = dataframe.drop(columns=['mol'])

        
        ### FIXING DATATYPES IN ODER TO MAKE PROPER ROUNDING ###
        dataframe["XG_Drug"] = dataframe["XG_Drug"].to_numpy(dtype=np.float64)
        dataframe["XG_Herbicide"] = dataframe["XG_Herbicide"].to_numpy(dtype=np.float64)
        ### ROUND ###
        dataframe = dataframe.round(decimals =roundby)

        print(type(dataframe["XG_Drug"][0]))
        ### SHOW TABLE ####
        if InputSelector == "Smiles String":
            st.dataframe(dataframe)
            
            

        elif InputSelector == "CSV":
            AgGrid(dataframe)

  
        if  VisualizationSelector == "Yes":
            st.write(
            """
            ## Molecule Visualizer
            
            """)
            st.markdown("Classified  with threshold: " + str(Threshold))
            st.markdown(DrugLabel  + ": Drug")
            st.markdown(HerbicideLabel + ": Herbicide")


            XG_herbicide = dataframe[["XG_Herbicide"]].iloc[:, 0].tolist()
            if type(user_input) == str:
                    if dataframe["XG_Drug"][0] >= Threshold:
                        Label = DrugLabel 
                    else:
                        Label = HerbicideLabel


                    cols = st.columns([1,1,6,1,1])
                    for i in range(0,len(cols)):
                            with cols[i]:
                                    if i == 2:
                                            out =  f"                               XG herbicide: {round(XG_herbicide[0],4)}"
                                            st.write(out)
                                            m = Chem.MolFromSmiles(user_input)
                                            im= Draw.MolToImage(m)
                                            st.image(im,caption=Label+user_input)
                                    else:
                                            pass

            elif  type(user_input) == list:
                    n = dataframe.shape[0]
                    
                    captions = []
                    for i,a in zip(range(0,n),user_input):
                        if dataframe["XG_Drug"][i] >= Threshold:
                            Label = DrugLabel 
                        else:
                            Label = HerbicideLabel
                        
                        captions.append(Label+a)


                    
                    mols = [Chem.MolFromSmiles(i) for i in user_input] 
                    ims  = [Draw.MolToImage(i) for i in mols]
                    image_iterator = paginator("Molecules", ims)
                    indices_on_page, images_on_page = map(list, zip(*image_iterator))
                    st.image(images_on_page, width=150, caption=captions)
        else:
            pass