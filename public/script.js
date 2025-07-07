let modelContainer = document.getElementById("container");
let model = new TSP.models.Sequential(modelContainer);

model.load( {
    type: "tfjs",
    url: "./jtfs_model/model.json"
} );

model.init();