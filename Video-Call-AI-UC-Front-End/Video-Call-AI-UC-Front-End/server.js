// Express.js code
const express = require('express');
const cors = require('cors'); // Import the cors module
const axios = require('axios');
const bodyParser = require('body-parser');
const pino = require('express-pino-logger')();

const app = express();
app.use(cors()); // Use the cors middleware to allow all cross-origin requests
app.use(bodyParser.urlencoded({ extended: false }));
app.use(pino);

app.get('/api/get-speech-token', async (req, res, next) => {
    res.setHeader('Content-Type', 'application/json');
    const speechKey = 'aed6a2cbcbd142fa839aeae67de535a4'; // For development environment only, remember to remove in production
    const speechRegion = 'australiaeast'; // For development environment only, remember to remove in production

    if (!speechKey || !speechRegion) {
        res.status(400).send('You forgot to add your speech key or region to the .env file.');
    } else {
        const headers = { 
            headers: {
                'Ocp-Apim-Subscription-Key': speechKey,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        };

        try {
            const tokenResponse = await axios.post(`https://${speechRegion}.api.cognitive.microsoft.com/sts/v1.0/issueToken`, null, headers);
            res.send({ token: tokenResponse.data, region: speechRegion });
        } catch (err) {
            res.status(401).send('There was an error authorizing your speech key.');
        }
    }
});

app.listen(3001, () =>
    console.log('Express server is running on localhost:3001')
);
