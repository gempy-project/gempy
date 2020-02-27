import os
import requests
import json
import datetime
package_directory = os.path.dirname(os.path.abspath(__file__))


class RexAPI:

    def __init__(self):
        self.api_key = None
        self.token_ID = None
        self.secret = None
        self.owner = None
        self.access_token = None
        self.project_urn = None
        self.project_link = None
        self.root_reference = None
        self.root_reference_link = None
        self.file_reference_link = None
        self.project_file_urn = None
        self.project_file_uplink = None

        self.response = None  # saves the most current server response for debugging

    def read_credentials(self, filename = os.path.join(package_directory, 'RexCloud_Api_key.txt')):
        with open(filename, "r") as file:
            self.token_ID = file.readline().strip('\n')
            self.secret = file.readline().strip('\n')

    def authorize_session(self):

        headers = {'Accept': 'application/json;charset=UTF-8',
                   'Content-Type': 'application/x-www-form-urlencoded; charset=ISO-8859-1'}

        data = {'grant_type': 'client_credentials'}

        self.response = requests.post('https://rex.robotic-eyes.com/oauth/token',
                                      headers=headers, data=data, auth=(self.token_ID, self.secret)
                                      )
        if self.response.status_code == 200:
            self.access_token = self.response.json()['access_token']

        else:
            print("something went wrong! Status code: "+str (self.response.status_code))

    def get_user_information(self):
        headers = {'Authorization': 'Bearer ' + self.access_token, 'Accept': 'application/json;charset=UTF-8'}
        self.response = requests.get('https://rex.robotic-eyes.com/api/v2/users/current', headers=headers)

        if self.response.status_code == 200:
            self.owner = self.response.json()['userId']

        else:
            print("something went wrong! Status code: "+str (self.response.status_code))

    def create_project(self, project_name):
        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'Accept': 'application/json;charset=UTF-8',
            'Content-Type': 'application/json;charset=UTF-8'
        }

        data = json.dumps({"name" : project_name,  "owner" : self.owner})  # this call expects json!

        self.response = requests.post('https://rex.robotic-eyes.com/api/v2/projects', headers=headers, data=data)

        if self.response.status_code == 201:
            self.project_urn = self.response.json()['urn']
            self.project_link = self.response.json()['_links']['self']['href']

        else:
            print("something went wrong! Status code: " + str(self.response.status_code))

    def create_root_reference(self):
        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'Accept': 'application/json;charset=UTF-8',
            'Content-Type': 'application/json;charset=UTF-8',
        }

        data = json.dumps({"project" : self.project_link,
                "name" : "root reference",
                "address" : {   "addressLine1" : "Sample",    "postcode" : "52072",
                                "city" : "Aachen",   "country" : "Austria"  }}
                          )

        self.response = requests.post('https://rex.robotic-eyes.com/api/v2/rexReferences', headers=headers, data=data)
        if self.response.status_code == 201:
            self.root_reference_link = self.response.json()['_links']['self']['href']

        else:
            print("something went wrong! Status code: " + str(self.response.status_code))

    def create_file_ressource_reference(self):
        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'Accept': 'application/json;charset=UTF-8',
            'Content-Type': 'application/json;charset=UTF-8',
        }

        data = json.dumps({"project" : self.project_link,
                           "name" : "root reference",
                           "parentReference" : self.root_reference_link})

        self.response = requests.post('https://rex.robotic-eyes.com/api/v2/rexReferences', headers=headers, data=data)
        if self.response.status_code == 201:
            self.file_reference_link = self.response.json()['_links']['self']['href']

        else:
            print("something went wrong! Status code: " + str(self.response.status_code))


    def create_project_file(self, filename):
        headers = {
            'Authorization':'Bearer ' + self.access_token,
            'Accept': 'application/json;charset=UTF-8',
            'Content-Type': 'application/json;charset=UTF-8',
                 }

        data = json.dumps({"project" : self.project_link,
                           "name" : filename,
                           "type" : "rex",
                           "rexReference" : self.file_reference_link
                           })

        self.response = requests.post('https://rex.robotic-eyes.com/api/v2/projectFiles', headers=headers, data=data)

        if self.response.status_code == 201:
            self.project_file_urn = self.response.json()['urn']
            self.project_file_uplink = self.response.json()['_links']['file.upload']['href']

        else:
            print("something went wrong! Status code: " + str(self.response.status_code))

    def upload_rexfile(self, filename):

        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'contentType': 'application/octet-stream',
            'type' : 'rex'
        }

        file = {
            'file':  (filename, open(filename, 'rb')),

        }

        self.response = requests.post(self.project_file_uplink,
                                 headers=headers, files=file)

    def get_project_files(self):

        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'Accept': 'application/json;charset=UTF-8'
            }

        self.response = requests.get(self.project_link + '/projectFiles', headers=headers)


    def upload_rexfiles(self, infiles : list):
        """
        wrapper around api calls to upload rexfiles of a gempy model.

        you will need to register an account under https://app.rexos.cloud/ .
        create an api key and store the key and secret in RexCloud_Api_key.txt.

        the function will take a list of rexos input filenames and uploads them into a newly created project.

        an ar code is plotted that can be scanned with the rexview app to show the model in vr.

        all api calls are python implementation of the Rex os api:
        https://www.rexos.org/rex-api/#tutorial-rex-project-before-you-begin


        Args:
            infiles: List of rexos file names

        Returns:

        """

        self.read_credentials()
        self.authorize_session()
        self.get_user_information()

        timestamp = datetime.datetime.now()
        self.create_project(str(timestamp))
        self.create_root_reference()

        for file in infiles:
            self.create_file_ressource_reference()
            self.create_project_file(file)
            self.upload_rexfile(file)

        #TODO: get tag, display tag




