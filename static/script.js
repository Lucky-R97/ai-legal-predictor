
async function predict(){

let q=document.getElementById("question").value

let res=await fetch("/predict",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({question:q})
})

let data=await res.json()

document.getElementById("result").innerText="Prediction: "+data.prediction

}
