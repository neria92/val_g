var wms_layers = [];


        var lyr_OpenStreetMap_0 = new ol.layer.Tile({
            'title': 'OpenStreetMap',
            'type': 'base',
            'opacity': 1.000000,
            
            
            source: new ol.source.XYZ({
    attributions: ' ',
                url: 'http://a.tile.openstreetmap.org/{z}/{x}/{y}.png'
            })
        });
var format_CuiliacnMisiones_1 = new ol.format.GeoJSON();
var features_CuiliacnMisiones_1 = format_CuiliacnMisiones_1.readFeatures(json_CuiliacnMisiones_1, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_CuiliacnMisiones_1 = new ol.source.Vector({
    attributions: ' ',
});
jsonSource_CuiliacnMisiones_1.addFeatures(features_CuiliacnMisiones_1);
var lyr_CuiliacnMisiones_1 = new ol.layer.Vector({
                declutter: true,
                source:jsonSource_CuiliacnMisiones_1, 
                style: style_CuiliacnMisiones_1,
                interactive: true,
    title: 'Cuiliacán Misiones<br />\
    <img src="styles/legend/CuiliacnMisiones_1_0.png" /> Registrada<br />\
    <img src="styles/legend/CuiliacnMisiones_1_1.png" /> Sin registro previo<br />'
        });

lyr_OpenStreetMap_0.setVisible(true);lyr_CuiliacnMisiones_1.setVisible(true);
var layersList = [lyr_OpenStreetMap_0,lyr_CuiliacnMisiones_1];
lyr_CuiliacnMisiones_1.set('fieldAliases', {'Id': 'Id', 'Store': 'Store', 'Name': 'Name', 'Address': 'Address', 'Latitude': 'Latitude', 'Longitude': 'Longitude', 'Date': 'Date', 'User_Latitude': 'User_Latitude', 'User_Longitude': 'User_Longitude', 'Url_Photo': 'Url_Photo', 'Url_Video': 'Url_Video', 'Nuevas': 'Nuevas', });
lyr_CuiliacnMisiones_1.set('fieldImages', {'Id': 'Range', 'Store': 'TextEdit', 'Name': 'TextEdit', 'Address': 'TextEdit', 'Latitude': 'TextEdit', 'Longitude': 'TextEdit', 'Date': 'TextEdit', 'User_Latitude': 'TextEdit', 'User_Longitude': 'TextEdit', 'Url_Photo': 'TextEdit', 'Url_Video': 'TextEdit', 'Nuevas': 'TextEdit', });
lyr_CuiliacnMisiones_1.set('fieldLabels', {'Id': 'inline label', 'Store': 'inline label', 'Name': 'inline label', 'Address': 'inline label', 'Latitude': 'inline label', 'Longitude': 'inline label', 'Date': 'inline label', 'User_Latitude': 'inline label', 'User_Longitude': 'inline label', 'Url_Photo': 'inline label', 'Url_Video': 'inline label', 'Nuevas': 'inline label', });
lyr_CuiliacnMisiones_1.on('precompose', function(evt) {
    evt.context.globalCompositeOperation = 'normal';
});