const express = require('express')
const { PythonShell } = require('python-shell');
const upload = require('./controller/uploader');
const fs = require('fs')
const spawn = require('child_process').spawn;


const app = express()
const port = 3000

app.set('view engine', 'ejs');
app.use(express.static("public"));

let excelFilePath = null
let _myState = false
let _myJson = null

app.post('/result', upload.single('file'), (req, res, next) => {
    excelFilePath = req.file.filename
    let optionData = "TOLL"
    let pyProgram = new PythonShell('./index.py', { terminateOnEnd: true, args: [excelFilePath, optionData] });
    pyProgram.on('message', function(message) {
        _myJson = message
        console.log(_myJson);
    });
    pyProgram.end(function(err, code, signal) {
        if (err) throw err;
        _myState = true

    });
    console.log(process.argv[2]);
    res.render('viewResult', {
        title: 'Plant-Science',
        message: 'Express.js example',
        myState: _myState
    });
});
app.get('/result', (req, res) => {
    let optionData = req.query.myselect;

    let pyProgram = new PythonShell('index.py');
    pyProgram.on('message', function(message) {
        console.log(message);
    });
    PythonShell.run('./index.py', { args: [excelFilePath, optionData] }, function(err, results) {
        if (err) throw err;
        console.log('Python script returned:', results);
        _myState = true
    });
    res.render('viewResult', {
        title: 'Plant-Science',
        message: 'Express.js example',
        myState: _myState
    });
});
app.get('/', (req, res) => {
    res.render('viewIndex', {
        title: 'Plant-Science',
        message: 'Express.js example',
    });
})

app.listen(port, () => console.log(`app listening on port ${port}!`))