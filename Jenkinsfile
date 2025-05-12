pipeline{
    agent any

    environment{
        VENV_DIR = 'venv'
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
                    - ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
    }
}