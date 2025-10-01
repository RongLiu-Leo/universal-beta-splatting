document.addEventListener('DOMContentLoaded', domReady);
    let dicsStatic = null;
    let dicsLight = null;
        function domReady() {
            dicsStatic = new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            dicsLight = new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"

            });
        }

        function staticSceneEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.static')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];
        
                let parts = media.src.split('/');
        
                switch (idx) {
                    case 0:
                        parts[parts.length - 2] = 'counter';
                        break;
                    case 1:
                        parts[parts.length - 2] = 'bicycle';
                        break;
                }
        
                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }

            let scene_list = document.getElementById("static-reconstruction").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dicsStatic.medias = dicsStatic._getMedias();
        }

        function sceneEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.light')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];
        
                let parts = media.src.split('/');
        
                switch (idx) {
                    case 0:
                        parts[parts.length - 3] = 'counter';
                        break;
                    case 1:
                        parts[parts.length - 3] = 'garden';
                        break;
                    case 2:
                        parts[parts.length - 3] = 'kitchen';
                        break;
                }
        
                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }

            let scene_list = document.getElementById("scene-selection").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dicsLight.medias = dicsLight._getMedias();
        }

        function lightSceneEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.light')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];
        
                let parts = media.src.split('/');
        
                switch (idx) {
                    case 0:
                        parts[parts.length - 1] = 'Diffuse.mp4';
                        break;
                    case 1:
                        parts[parts.length - 1] = 'RGB.mp4';
                        break;
                    case 2:
                        parts[parts.length - 1] = 'Specular.mp4';
                        break;
                }
        
                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }

            let scene_list = document.getElementById("light-decomposition").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dicsLight.medias = dicsLight._getMedias();
        }

