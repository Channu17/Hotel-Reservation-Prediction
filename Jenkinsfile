pipeline{
    agent any

    stages{
        stage('Cloning Github repo to jenkins'){
            steps{
                script{
                    echo 'cloning the gtihub repo to jenkins'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Channu17/Hotel-Reservation-Prediction.git']])
                }
            }
        }
    }
}