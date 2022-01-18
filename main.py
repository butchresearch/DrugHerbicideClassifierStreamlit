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
import base64 


@st.cache 
def convert_df(df):
    compression_opts = dict(method='zip',archive_name='out.csv')  
    return df.to_csv(compression=compression_opts).encode('utf-8')
@st.cache 
def get_table_download_link_csv(df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="DHC.csv" target="_blank">Download csv file</a>'
    return href

st.set_page_config(page_title="Drug Herbicide Classifier", page_icon ="💠", layout="wide")
DrugLabel      = "💊"
HerbicideLabel = "🌿"
predictions    = ["XG_Drug","XG_Herbicide","LR_Drug","LR_Herbicide","RF_Drug","RF_Herbicide","SVM_Drug","SVM_Herbicide"]
dataframe      = pd.DataFrame()
user_input     = None
model          = "LR_Drug"
st.title('Drug Herbicide Classifier: Drug Chemical Space as a Guide for New Herbicide Development: A Cheminformatic Analysis ') #Set Tittle
st.markdown(""" 
**BACKGROUND:** 

Herbicides are critical resources for meeting agricultural demand. While 
similar in structure and function to pharmaceuticals, the development of new herbicidal 
mechanisms of action and new scaffolds against known mechanisms of action has been 
much slow compared to pharmaceutical sciences. 

**RESULTS:**

 We trained several machine learning techniques to classify herbicides versus 
drugs based on physicochemical characteristics. The most accurate, has an accuracy of 
93%. The key differentiating characteristics were polar hydrogens, number of amide 
bonds, solubility, and polar surface area. We then analyzed the diversity of each set 
based on scaffolds and scaffold decomposition and showed the chemical diversity of 
herbicides to be considerably lower. Finally, we conducted docking assays with 
herbicides modified with complementary structural components only present in drugs, 
and show the increased chemical diversity to enhance herbicide binding to enzyme 
targets. 

**CONCLUSION:**

 Herbicides are distinct from drugs based on physicochemical properties, 
but less diverse in their chemistry in a way not governed by these properties. Increasing 
the diversity of herbicide scaffolds has the potential to increase potency, potentially 
reducing the amount needed in agricultural practice.""")
st.write(
        """
        ## Explore our Models Online
        
        """)
st.markdown("This is a compendium of the Machine Learning models trained at Drug Chemical Space as a Guide for New Herbicide Development: A Cheminformatic Analysis")

st.markdown("## How to use it?")
st.markdown("Using the menu sidebar select the parameters you wish to use (If the sidebar is not visable click on the arrow a the top left of your screen)")
st.markdown("1. Select Input type")
st.markdown("2. Parse the Data")
st.markdown("3. Select if you wish to visualize molecules ")
st.markdown("4. Select output (Simplified only displays predictions, verbose displays additional metadata use during tryning) ")
st.markdown("5. Select the model you wish to use for visualization (This loads the model and the optimal trheshold) ")
st.markdown("6. If you wish to manually select another threshold click on 'Yes' on Customize Thresshold ")
st.markdown("7. A new slide will appear, select the threshold value (0.5 by default)")
st.markdown("8. Select to wich decimal place you wish to round the output")
st.markdown("9. For csv input if the number of moleucles exceedes the maximum of display port you can switch to a new page using the Molecules page")
def paginator(label, items, items_per_page=27, on_sidebar=True):
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
        st.markdown("Write a valid SMILES STRINGS in the text section")
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
OutputSelector        = st.sidebar.selectbox("Select Outpuy Type",("Simplified","Verbose"))                     # Select input Type 
ModelSelector         = st.sidebar.selectbox("Select Model for Visualization",("Linear Regression","Random Forest","Support Vector Machine","Xgboost"))

if ModelSelector == "Linear Regression":
    model       = "LR_Drug"
    Threshold   =  0.59
elif ModelSelector == "Random Forest":
    model       = "RF_Drug"
    Threshold   =  0.21
elif ModelSelector == "Support Vector Machine":
    model = "SVM_Drug"
    Threshold   =  0.19
elif ModelSelector == "Xgboost":
    model = "XG_Drug"
    Threshold   =  0.08
CustomTrheshold       = st.sidebar.radio("Customize Threshold",('Yes', 'No'),index=1)     
if CustomTrheshold == "Yes":
    Threshold             = st.sidebar.slider("Threshold",min_value=0.1, max_value=0.9,value=0.5)
roundby               = st.sidebar.slider("Decimal Places",min_value=1, max_value=6,value=6)





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
            st.markdown(f"Classified Model: {ModelSelector} with threshold: " + str(Threshold))
            st.markdown(f"Lables for molecule predictions:")
            st.markdown(f"* Drug Lable: {DrugLabel}")
            st.markdown(f"* Herbicide: {HerbicideLabel}")


            XG_herbicide = dataframe[["XG_Herbicide"]].iloc[:, 0].tolist()
            if type(user_input) == str:
                    if dataframe[model][0] >= Threshold:
                        Label = DrugLabel 
                    else:
                        Label = HerbicideLabel


                    cols = st.columns([1,1,6,1,1])
                    for i in range(0,len(cols)):
                            with cols[i]:
                                    if i == 2:
                                            #out =  f"                               XG herbicide: {round(XG_herbicide[0],4)}"
                                            #st.write(out)
                                            m = Chem.MolFromSmiles(user_input)
                                            im= Draw.MolToImage(m)
                                            st.image(im,caption=Label+user_input)
                                    else:
                                            pass

            elif  type(user_input) == list:
                    csv = convert_df(dataframe)
                    st.download_button('📥 Download Current Result',csv,"DHC.csv","text/csv",key='download-csv')
                    #st.markdown(get_table_download_link_csv(dataframe), unsafe_allow_html=True)
                    n = dataframe.shape[0]
                    
                    captions = []
                    for i,a in zip(range(0,n),user_input):
                        if dataframe[model][i] >= Threshold:
                            Label = DrugLabel 
                        else:
                            Label = HerbicideLabel
                        
                        captions.append(Label+a)


                    
                    mols = [Chem.MolFromSmiles(i) for i in user_input] 
                    ims  = [Draw.MolToImage(i) for i in mols]
                    image_iterator = paginator("Molecules", ims)
                    indices_on_page, images_on_page = map(list, zip(*image_iterator))
                    current_captions = [captions[i] for i in indices_on_page]           # Retrive Current Captions
              
                    st.image(images_on_page, width=150,caption=current_captions)


        else:
            pass