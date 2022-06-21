import streamlit as st
import pickle
import numpy as np
model_A=pickle.load(open('model_A.pkl','rb'))
model_F=pickle.load(open('model_F.pkl','rb'))



def predict_F(Age,Alpha2_Macroglobuline,Haptoglobine,Apolipoproteine_A1, Bilirubine,Gamma_GT,ALAT,Sexe_Femme, Sexe_Homme):
    input=np.array([[Age,Alpha2_Macroglobuline,Haptoglobine,Apolipoproteine_A1, Bilirubine,Gamma_GT,ALAT,Sexe_Femme, Sexe_Homme]]).astype(np.float64)
    prediction_F=model_F.predict(input)
    
    pred_F='{0:.{1}f}'.format(prediction_F[0][0], 2)
    
    return float(pred_F)

def predict_A(Age,Alpha2_Macroglobuline,Haptoglobine,Apolipoproteine_A1, Bilirubine,Gamma_GT,ALAT,Sexe_Femme, Sexe_Homme):
    input=np.array([[Age,Alpha2_Macroglobuline,Haptoglobine,Apolipoproteine_A1, Bilirubine,Gamma_GT,ALAT,Sexe_Femme, Sexe_Homme]]).astype(np.float64)
    prediction_A=model_A.predict(input)
    
    pred_A='{0:.{1}f}'.format(prediction_A[0][0], 2)
    
    return float(pred_A)

def main():
    st.title("Prédiction médicale")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Actitest, Fibrotest </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Age = st.text_input("Age",68)
    Alpha2_Macroglobuline = st.text_input("Alpha2_Macroglobuline",1.94)
    Haptoglobine = st.text_input("Haptoglobine",1.19)
    Apolipoproteine_A1 = st.text_input("Apolipoproteine_A1",1.42)
    Bilirubine = st.text_input("Bilirubine",5)
    Gamma_GT = st.text_input("Gamma_GT",22)
    ALAT = st.text_input("ALAT",11)
    Sexe_Femme = st.text_input("Sexe_Femme",0)
    Sexe_Homme = st.text_input("Sexe_Homme",1)


    A0_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> A0</h2>
       </div>
    """
    A0_A1_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> A0-A1</h2>
       </div>
    """

    A1_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> A1</h2>
       </div>
    """
    A2_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> A2</h2>
       </div>
    """

    A3_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> A3</h2>
       </div>
    """

    F0_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> F0</h2>
       </div>
    """
    F0_F1_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> F0-F1</h2>
       </div>
    """

    F1_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> F1</h2>
       </div>
    """
    F2_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> F2</h2>
       </div>
    """

    F3_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> F3</h2>
       </div>
    """

    F4_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> F4</h2>
       </div>
    """

    if st.button("Predict"):
        output_A=predict_A(Age,Alpha2_Macroglobuline,Haptoglobine,Apolipoproteine_A1, Bilirubine,Gamma_GT,ALAT,Sexe_Femme, Sexe_Homme)
        st.success('Actitest value {}'.format(output_A))
        output_F=predict_F(Age,Alpha2_Macroglobuline,Haptoglobine,Apolipoproteine_A1, Bilirubine,Gamma_GT,ALAT,Sexe_Femme, Sexe_Homme)
        st.success('Fibrotest value {}'.format(output_F))

        if output_A < 0.25:
            st.markdown(A0_html,unsafe_allow_html=True)
        elif output_A < 0.31 :
            st.markdown(A0_A1_html,unsafe_allow_html=True)
        elif output_A < 0.51 :
            st.markdown(A1_html,unsafe_allow_html=True)
        elif output_A < 0.625 :
            st.markdown(A2_html,unsafe_allow_html=True)
        else: 
            st.markdown(A3_html,unsafe_allow_html=True)

        if output_F < 0.25:
            st.markdown(F0_html,unsafe_allow_html=True)
        elif output_F < 0.31 :
            st.markdown(F0_F1_html,unsafe_allow_html=True)
        elif output_F < 0.5 :
            st.markdown(F1_html,unsafe_allow_html=True)
        elif output_F < 0.6 :
            st.markdown(F2_html,unsafe_allow_html=True)
        elif output_F < 0.75 :
            st.markdown(F3_html,unsafe_allow_html=True)
        else: 
            st.markdown(F4_html,unsafe_allow_html=True)


if __name__=='__main__':
    main()