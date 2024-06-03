import streamlit as st
import requests
import pandas as pd
def post(payload, t):
    url = f'http://fastapi:80/post_todos?tipe={t}'
    response = requests.post(url, json=payload)
    # Handle the response
    if response.status_code == 200:
        data = response.json()
    else:
        st.write("Error:", response.status_code)
        st.write(response.text)

def get_all():
    url = 'http://fastapi:80/get_todos'
    response = requests.get(url)
    data = response.json()
    df = []
    info = False
    for todo in data:
        info=True
        df.append({"ID": todo[0], "Todo": todo[1], "Type": todo[2], "is_widget": False})
    df = pd.DataFrame(df)
    edited_df = st.data_editor(df)
    return (edited_df, info)

def delete(id_list, todo_list):
    for i in range(len(id_list)):
        url = f'http://fastapi:80/delete?id={id_list[i]}&task={todo_list[i]}'
        response = requests.delete(url)

    

col1, col2, col3 = st.columns(3)
id = col1.text_input("Id")
task = col2.text_input("Task")
t = col3.selectbox(
    "Type",
   ("physical", "mental", "busy"),
   index=None,
   placeholder="Select Type...",
)

if st.button("Submit Todo"):
    if id.isnumeric():
        id = int(id)
        payload = {
            "id": id,
            "todo": task,
            "typeTodo": t
        }
        post(payload, t)
    else:
        st.write("ID should be an integer")
edited_df, not_empty = get_all()
id_list = []
todo_list = []
if not_empty:
    completed = edited_df.loc[edited_df["is_widget"] == True]
    for i in completed["ID"]:
        id_list.append(i)
    for i in completed["Todo"]:
        todo_list.append(i)
if st.button("Delete Completed Tasks") and id_list:
    delete(id_list, todo_list)
    st.experimental_rerun()



