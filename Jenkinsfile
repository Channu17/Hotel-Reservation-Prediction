pipeline{
    agent any

    environment{
        VENV_DIR = 'venv'
        GCP_PROJECT = "plexiform-guide-458805-j8"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
    }

    stages{
        stage('Cloning Github repo to jenkins'){
            steps{
                script{
                    echo 'cloning the gtihub repo to jenkins'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Channu17/Hotel-Reservation-Prediction.git']])
                }
            }
        }
        stage('Creating virtual environment'){
            steps{
                script{
                    echo 'creating virtual environment'
                    sh '''
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
        stage('Building and pushing Docker image to GCR'){
            steps{
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script{
                        echo 'Building and pushing Docker image to GCR'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/hotel-reservation-prediction:latest .
                        docker push gcr.io/${GCP_PROJECT}/hotel-reservation-prediction:latest
                        '''
                    }
                }
            }
        }
        stage('Deploy to Google cloude run'){
            steps{
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script{
                        echo 'Deploy to Google cloude run'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        
                        gcloud run deploy hotel-reservation-prediction \
                        --image=gcr.io/${GCP_PROJECT}/hotel-reservation-prediction:latest \
                        --platform=managed \
                        --region=us-central1 \
                        --allow-unauthenticated
                        '''
                    }
                }
            }
        }
    }
}