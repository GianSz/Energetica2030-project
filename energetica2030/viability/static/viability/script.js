let selectVechile = document.getElementById('selectVehicle');
selectVechile.addEventListener('change', ()=>{
    removeOptions();
    addOptions();
});

/**
 * This function is in charge of removing the options that were added when a change occurred to the select element 
 * that has 'selectVechicle' as id attribute.
 * @returns null in case the select vehicle is not an correct option
 */
function removeOptions(){
    let selectType = document.getElementById('selectType');
    if(selectType.children.length == 3){
        selectType.lastElementChild.remove();
        selectType.lastElementChild.remove();
    }
}

/**
 * This function is in charge of creating two option elements inside a select element that has 'selectType' as id attribute.
 * The first option element always have the text 'Combustión' inside of it no matter what. On the other hand, the second 
 * option element text depends on the previosly chosen vehicle (which was chosen using another select element), therefore, if
 * the chosen vehicle was 'moto', then the text will be 'Híbrida', by contrast, if the chosen vehicle was 'lancha', then the
 * text will be 'Eléctrica'.
 * @returns null in case the select vehicle is not an correct option
 */
function addOptions(){
    if(selectVechile.value == '0') return null;

    let selectType = document.getElementById('selectType');
    let frag = document.createDocumentFragment();
    let option1 = document.createElement('option');
    option1.value = '1';
    let option2 = document.createElement('option');
    option2.value = '2';

    option1.innerHTML = 'Combustión';
    if(selectVechile.value == '1') option2.innerHTML = 'Híbrida';
    else option2.innerHTML = 'Eléctrica';

    frag.appendChild(option1);
    frag.appendChild(option2);
    selectType.appendChild(frag);
}