from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_valid_importDocument():
    file_path = "./Test_Document.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        modelName = {"modelName": "llama3"}
        # Perform the request
        response = client.post("/importDocument/", params = modelName, files=file)
        # Assertions
        json_response = response.json()
        print(json_response)
        assert response.status_code == 200
        assert json_response["Message"] == "Sucessfully added document Test_Document.pdf"

#the test above alread imported the document, so here we import the same document again
def test_duplicate_importDocument():
    file_path = "./Test_Document.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        modelName = {"modelName": "llama3"}
        # Perform the request
        response = client.post("/importDocument/", params = modelName, files=file)
        # Assertions
        json_response = response.json()
        assert response.status_code == 200
        assert json_response["message"] == "llama3 has file Test_Document already imported"

def test_invalidModel_importDocument():
    file_path = "./Test_Document.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        modelName = {"modelName": "foo"}
        # Perform the request
        response = client.post("/importDocument/", params = modelName, files=file)
        # Assertions
        json_response = response.json()
        assert response.status_code == 200
        assert json_response["Message"] == "Model foo not found"

def test_invalidDocument_importDocument():
    file_path = "./dummy_dataset.csv"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "text/csv")}
        modelName = {"modelName": "llama3"}
        # Perform the request
        response = client.post("/importDocument/", params = modelName, files=file)
        # Assertions
        json_response = response.json()
        assert response.status_code == 200
        assert json_response["Message"] == "Please import a PDF"

def test_invalidDoc_delete_document():
    params = {"document": "foo", "modelName": "llama3"}
    # Perform the request
    response = client.delete("/delete_document/", params = params)
    # Assertions
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["message"] == "llama3 does not have document foo imported"

def test_invalidModel_delete_document():
    params = {"document": "Test_Document", "modelName": "bar"}
    # Perform the request
    response = client.delete("/delete_document/", params = params)
    # Assertions
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["Message"] == "Model bar not found"

def test_valid_delete_document():
    params = {"document": "Test_Document", "modelName": "llama3"}
    response = client.delete("/delete_document/", params = params)
    # Assertions
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["message"] == "Successfully deleted document Test_Document from model llama3"


def test_valid_list_documents():
    file_path = "./Test_Document.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        modelName = {"modelName": "llama3"}
        # Perform the request
        client.post("/importDocument/", params = modelName, files=file)
    file_path = "./Test_Document2.pdf"
    with open(file_path, "rb") as file:
        file = {"file": (file_path.split("/")[-1], file, "application/pdf")}
        # Perform the request
        client.post("/importDocument/", params = modelName, files=file)
    response = client.get("/list_documents/", params = modelName)
    json_response = response.json()
    # print(json_response)
    assert response.status_code == 200
    print(json_response["Imported Documents for llama3"])
    assert json_response["Imported Documents for llama3"] == ["Test_Document2", "Test_Document"]

def test_invalidModel_deleteAll_Documents():
    params = {"modelName": "foobar"}
    response = client.delete("/deleteAll_Documents/", params = params)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["Message"] == "Model foobar not found"

def test_valid_deleteAll_Documents():
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



