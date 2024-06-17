import os
from fastapi.testclient import TestClient
import sys
sys.path.append('/code/app')
from main import app
client = TestClient(app)

os.chdir("/code")

def test_valid_importDocument():
    file_path = "./app/testing_folder/Test_Document.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        modelName = {"modelName": "llama3"}
        response = client.post("/importDocument/", params = modelName, files=file)
        json_response = response.json()
        print(json_response)
        assert response.status_code == 200
        assert json_response["Message"] == "Sucessfully added document Test_Document.pdf"

#the test above alread imported the document, so here we import the same document again
def test_duplicate_importDocument():
    file_path = "./app/testing_folder/Test_Document.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        modelName = {"modelName": "llama3"}
        response = client.post("/importDocument/", params = modelName, files=file)
        json_response = response.json()
        assert response.status_code == 200
        assert json_response["message"] == "llama3 has file Test_Document already imported"

def test_invalidModel_importDocument():
    file_path = "./app/testing_folder/Test_Document.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        modelName = {"modelName": "foo"}
        response = client.post("/importDocument/", params = modelName, files=file)
        json_response = response.json()
        assert response.status_code == 200
        assert json_response["Message"] == "Model foo not found"

def test_invalidDocument_importDocument():
    file_path = "./app/testing_folder/dummy_dataset.csv"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "text/csv")}
        modelName = {"modelName": "llama3"}
        response = client.post("/importDocument/", params = modelName, files=file)
        json_response = response.json()
        assert response.status_code == 200
        assert json_response["Message"] == "Please import a PDF"

def test_invalidDocument_deleteDocument():
    params = {"document": "foo", "modelName": "llama3"}
    response = client.delete("/delete_document/", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["message"] == "llama3 does not have document foo imported"

def test_invalidModel_deleteDocument():
    params = {"document": "Test_Document", "modelName": "bar"}
    response = client.delete("/delete_document/", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["Message"] == "Model bar not found"

def test_valid_deleteDocument():
    params = {"document": "Test_Document", "modelName": "llama3"}
    response = client.delete("/delete_document/", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["message"] == "Successfully deleted document Test_Document from model llama3"

def test_valid_listDocuments():
    file_path = "./app/testing_folder/Test_Document.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        modelName = {"modelName": "llama3"}
        client.post("/importDocument/", params = modelName, files=file)
    file_path = "./app/testing_folder/Test_Document2.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        client.post("/importDocument/", params = modelName, files=file)
    response = client.get("/list_documents/", params = modelName)
    json_response = response.json()
    assert response.status_code == 200
    print(json_response["Imported Documents for llama3"])
    assert set(json_response["Imported Documents for llama3"]) == set(["Test_Document2", "Test_Document"])

def test_invalidModel_deleteAllDocuments():
    params = {"modelName": "foobar"}
    response = client.delete("/deleteAll_Documents/", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["Message"] == "Model foobar not found"

def test_valid_deleteAllDocuments():
    params = {"modelName": "llama3"}
    response = client.delete("/deleteAll_Documents/", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["Message"] == "Successfully removed all documents for model llama3"

def test_duplicate_createModel():
    params = {"modelName": "llama3", "system": "", "baseModel": "llama3"}
    response = client.post(f"/createModel/llama3", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["Message"] == "Model Name llama3 already exists"

def test_valid_createModel():
    params =  {"modelName": "foobar", "system": "", "baseModel": "llama3"}
    response = client.post(f"/createModel/foobar", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["Message"] == "Successfully created foobar"

def test_list_models():
    response = client.get("/list_models")
    json_response = response.json()
    assert response.status_code == 200
    assert set(json_response["All Created Models"]) == set(["foobar", "llama3"])

def test_invalid_deleteModel():
    params = {"modelName": "foo_bar"}
    response = client.delete("/deleteModel/foo_bar", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["Message"] == "No model named foo_bar"
    assert set(json_response["Created Models"]) == set(["llama3", "foobar"])

def test_valid_deleteModel():
    params = {"modelName": "foobar"}
    response = client.delete("/deleteModel/foobar", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["Message"] == "Successfully deleted foobar"

def test_invalid_generate():
    params = {"user_prompt": "Hello", "modelName": "foo"}
    response = client.post("/generate", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["Message"] == "Model foo not found"


def test_valid_generate():
    params = {"user_prompt": "Hello", "modelName": "llama3"}
    response = client.post("/generate", params = params)
    assert response.status_code == 200

def test_validrRag_generate():
    file_path = "./app/testing_folder/Test_Document.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        modelName = {"modelName": "llama3"}
        
        client.post("/importDocument/", params = modelName, files=file)
    file_path = "./app/testing_folder/Test_Document2.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        
        client.post("/importDocument/", params = modelName, files=file)
    params = {"user_prompt": "Hello", "modelName": "llama3", "documents": ["Test_Document", "Test_Document2"]}
    response = client.post("/generate", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert "Test_Document.pdf" in json_response
    assert "Test_Document2.pdf" in json_response
    params = {"modelName": "llama3"}
    response = client.delete("/deleteAll_Documents/", params = params)

def test_twoInvalidFiles_finetune():
    file_path = "./app/testing_folder/Test_Document.pdf"
    with open(file_path, "rb") as file:
        files = {
            "train_dataset": (file_path.split("/")[-1], file, "application/pdf"),
            "eval_dataset": (file_path.split("/")[-1], file, "application/pdf")
        }
        response = client.post("fine_tune_model/foobar", files = files)
        json_response = response.json()
        assert response.status_code == 200
        assert json_response["Message"] == "Please import a CSV"

def test_oneInvalidFile_finetune():
    file_path1 = "./app/testing_folder/Test_Document.pdf"
    file_path2 = "./app/testing_folder/dummy_dataset.csv"
    with open(file_path1, "rb") as file1, open(file_path2, "rb") as file2:
        files = {
            "train_dataset": (file_path1.split("/")[-1], file1, "application/pdf"),
            "eval_dataset": (file_path2.split("/")[-1], file2, "application/pdf")
        }
        response = client.post("fine_tune_model/foobar", files = files)
        json_response = response.json()
        assert response.status_code == 200
        assert json_response["Message"] == "Please import a CSV"


def test_invalidName_finetune():
    file_path = "./app/testing_folder/dummy_dataset.csv"
    with open(file_path, "rb") as file:
        files = {
            "train_dataset": (file_path.split("/")[-1], file, "text/csv"),
            "eval_dataset": (file_path.split("/")[-1], file, "text/csv")
        }
        response = client.post("fine_tune_model/llama3", files = files)
        json_response = response.json()
        assert response.status_code == 200
        assert json_response["Message"] == "Model Name llama3 already exists"

def test_valid_finetune():
    file_path = "./app/testing_folder/dummy_dataset.csv"
    with open(file_path, "rb") as file:
        files = {
            "train_dataset": (file_path.split("/")[-1], file, "text/csv"),
            "eval_dataset": (file_path.split("/")[-1], file, "text/csv")
        }
        response = client.post("fine_tune_model/foobar", files = files)
        json_response = response.json()
        assert response.status_code == 200
        assert json_response["Message"] == "Sucessfully finetuned model foobar"
        params = {"modelName": "foobar"}
        response = client.delete("/deleteModel/foobar", params = params)