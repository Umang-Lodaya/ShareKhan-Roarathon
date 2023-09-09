import streamlit as st
from PIL import Image
import base64
import sqlite3
conn = sqlite3.connect('data.db',check_same_thread=False)
cur = conn.cursor()


def bookdonate():       
    st.markdown("<h1 style='text-align: left; color: yellow;'>Book Donation</h1>", unsafe_allow_html=True)
    
    st.title(" Welcome to the book donation page, your old book can bring light in someones future.\nCome let us donate books for needy one.\nYou don't have to walk and donate it you just have to register yourself and we will pick the book from your house address that will be provided.")
    st.markdown("## Here if you are willing to donate book\n you have to register yourself.")
    
    with st.form(key="Registration for Blood Donation"):
    

        bookname = st.text_input("Enter your name : ")
        address = st.text_input("Enter your address please : ")
        book_phone = st.text_input("Enter your phone Number : ")
        
        submissionbook = st.form_submit_button(label="Submit")
        if submissionbook==True:         
            addData(bookname,address,book_phone)
        else:
            st.info("Please submit the form.")
def addData(a,b,c):
    
    cur.execute("""CREATE TABLE IF NOT EXISTS book(NAME TEXT(50),ADDRESS TEXT(50), PHONE_NO  TEXT(15)); """) 
    cur.execute("INSERT INTO book VALUES (?,?,?)", (a,b,c))
    conn.commit()
    conn.close()
    st.success("Successfully inserted")
    

    