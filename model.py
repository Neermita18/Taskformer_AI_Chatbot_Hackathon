import os
from dotenv import load_dotenv
from datasets import load_dataset
from datasets import load_dataset
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np




ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
dataset1= [text['input']+" "+ text['output'] for text in ds]
df= pd.read_csv(r"C:\Users\91982\Desktop\Taskformer\data\All-2479-Answers-retrieved-from-MedQuAD.csv")

dataset2= [x for x in df['Answer']]
import os
import fitz
from PIL import Image
from io import BytesIO

# %%
def extract_images_and_descriptions(pdf_path, image_dir):
    os.makedirs(image_dir, exist_ok=True)
    document = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(document)):
        page = document[page_num]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_path = os.path.join(image_dir, f"image_{page_num+1}_{img_index+1}.{img_ext}")

            with open(img_path, "wb") as img_file:
                img_file.write(image_bytes)

            image_paths.append(img_path)
    return image_paths



pdf_path = r"C:\Users\91982\Desktop\Taskformer\data\dermet1.pdf"
image_dir =  r"C:\Users\91982\Desktop\Taskformer\images"
i=extract_images_and_descriptions(pdf_path, image_dir)

# %%
descriptions=["Comedonal acne with post-inflammatory hyperpigmentation", "Urticaria in a child", "Viral wart: earlobe", "Dermal naevus, skin type 5, macro 3",
              "Onychomycosis showing pigmentaton of the plate and subungal hyperkeratosis", "Amelanotic melanoma", "Halo mole", "Discoid eczema in an infant",
              "ice-pink scars over the cheeks", "blistering due to accidental spillage of hot cooking oil onto hand", "Comedonal acne", "Surrounding cellulitis",
              "Urticaria in skin of color", "An in-situ melanoma on the ear lobe", "Cutaneous squamous cell carcinoma in skin of color", "Well-defined scaly plaque psoriasis on knee", 
              "Acanthosis nigricans on the neck", "Facial and lid angioedema due to food allergy", "Honey-colored crusted lesions in facial impetigo", "fine pale scaly lesions of pityriasis versicolor",
              "infantile atopic dermatitis on hand", "early superficial plaques of psoriasis on back", "squamous cell carcinoma", "intraepidermal squamous cell carcinoma", "facial acne", "Keratoacanthoma face",
              "melanonychia", "small patch vitiligo over trunk", "tinea corporis on leg", "vitiligo affecting upper and lower lids symmetrically", "subepidermal calcified nodule on the nose", "basal cell carcinoma on face",
              "atopic hand dermatitis and follicular pattern on trunk", "two dark lesions on mid-upper back- both in situ melanomas", "atopic dermatitis","tense and resolving blisters of bullous pemphigoid on upper thigh, skin of color",
              "psoriasis of the scalp", "patchy eczema on the arm", "cyst", "vitiligo", "multiple lesions of calcinosis cutis on base of thumb", "left scapula", "clustered vesicles of herpes zoster infection in HIV", "A pigmented lesion on trunk with irregularity of colors; histology showed an in-situ melanoma", 
              "lichen planus in skin of color", "alopecia areata of the top of scalp", "conchal bowl discoid lupus erythematosus", "Superficial basal call carcinoma on leg"]
 
              


image_description_pairs = list(zip(i, descriptions))




actual_desc=[description for img_path, description in image_description_pairs]





import google.generativeai as genai
import PIL.Image


from dotenv import load_dotenv






    

# %%
ai_desc=[""" The image shows the upper back of a person with dark skin. The skin has numerous, widely scattered, small, dark bumps. This could be keratosis pilaris, acne, or folliculitis. 

The image shows the torso of a child with multiple, circular, erythematous, slightly raised lesions with central clearing. These lesions are most prominent on the abdomen and back. This appearance of the rash is suggestive of several possible skin conditions, including:

* **Tinea corporis (ringworm):** A fungal infection that often presents with ring-shaped rashes.
* **Nummular eczema:** A type of eczema that causes coin-shaped, itchy patches of skin.
* **Granuloma annulare:** A skin condition that causes raised, reddish or skin-colored bumps in a ring pattern.
* **Erythema multiforme:** An allergic reaction that can cause target-shaped lesions.
* **Urticaria (hives):** Raised, itchy welts on the skin that can be triggered by allergies or other factors. """, """The image shows a close-up view of a dark brown, raised lesion located on the earlobe. The lesion appears irregular in shape and texture.  Given its location and appearance, the following possibilities should be considered:

* **Seborrheic Keratosis:**  These are benign (non-cancerous) growths that are very common. They often appear waxy, stuck-on, and brown to black in color.
* **Melanocytic Nevus (Mole):** Moles can vary in appearance from flat to raised, and in color from tan to dark brown or black.  While most moles are benign, changes in their appearance can sometimes be a sign of skin cancer.
* **Dermatofibroma:** These are benign nodules that are firm to the touch and often have a dimple in the center. They are usually reddish-brown in color. 
* **Skin Tag (Acrochordon):** Skin tags are small, flesh-colored or brown growths that hang off the skin. They are benign and painless. 
* **Basal Cell Carcinoma:** While less likely in this location, basal cell carcinoma is a type of skin cancer that can present as a pearly or waxy bump.  """,
"""The image shows a close-up of a dark brown, irregularly shaped growth on the earlobe. 

**Possible diseases:**

* **Seborrheic keratosis:** These are benign (non-cancerous) growths that are often brown, black, or light tan. They can appear waxy, scaly, or slightly raised. 
* **Melanocytic nevus (mole):** Moles are common growths on the skin that can vary in color, size, and shape. While most moles are benign, changes in size, color, or shape could indicate melanoma (skin cancer).
* **Dermatofibroma:** These are harmless, firm nodules that are often brown, reddish, or pink. They typically develop on the legs or arms but can occur on the earlobe. 
* **Skin tag:** Skin tags are small, soft, flesh-colored growths that hang off the skin. While less common on the earlobe, they are possible. 
* **Melanoma:** Though less likely given the appearance, it's important to consider melanoma, a type of skin cancer, as a possibility, especially if the growth has changed recently. """,
"""The image shows a close-up view of a skin lesion located on the scalp. The lesion is raised, flesh-colored to pink, and has a rough, irregular surface with a cobblestone-like appearance. There is some surrounding erythema (redness).

Given its location and characteristics, possible diseases include:

* **Seborrheic Keratosis:** This is a benign (non-cancerous) growth that is very common, especially in older individuals. They often have a "stuck-on" appearance and a waxy or scaly surface.
* **Viral Wart (Verruca Vulgaris):** Warts can occur on the scalp and may have a rough, cauliflower-like appearance. They are caused by the human papillomavirus (HPV). 
* **Basal Cell Carcinoma:** While less likely given the appearance, this type of skin cancer can occur on the scalp and present as a pearly or waxy bump.
* **Squamous Cell Carcinoma:**  This is another type of skin cancer that can sometimes have a rough or wart-like texture. """, """The image shows a close-up view of a hand with dark, thickened, and leathery skin. The fingers appear swollen, and the nails are discolored and brittle. 

Possible conditions that could cause these symptoms include:

* **Scleroderma:** This autoimmune disease causes the body to produce too much collagen, leading to hardening and tightening of the skin. It can also affect internal organs.
* **Graft-versus-host disease:** This complication can occur after a bone marrow or stem cell transplant. The donor's immune cells attack the recipient's body, causing inflammation and damage to the skin, among other organs.
* **Chronic venous insufficiency:** This condition occurs when the veins in the legs and hands have trouble sending blood back to the heart. This can cause fluid buildup, swelling, and skin changes.
* **Acanthosis nigricans:** This skin condition causes dark, velvety patches of skin in body folds and creases, including the hands. It is often associated with insulin resistance and obesity.
* **Porphyria cutanea tarda:** This rare genetic disorder affects the production of heme, a component of red blood cells. It can cause blisters, scarring, and skin thickening on sun-exposed areas, including the hands.""", """The image shows a close-up view of the skin, potentially on the arm or torso. There is a single, well-defined, pink to red lesion present. It appears slightly raised and has a slightly darker center. 

Possible diagnoses based on this image include:

* **Dermatofibroma:** These are benign skin growths that often appear as firm, reddish-brown nodules.
* **Mole (Nevus):** Moles are common, benign skin growths that can vary in color, shape, and size.
* **Basal Cell Carcinoma:**  While less likely in a lesion of this appearance, it's important to consider as it can present with a reddish, slightly raised appearance. 
* **Merkel Cell Carcinoma:** This is a rare, aggressive skin cancer that can present as a firm, painless, flesh-colored or bluish-red nodule.""", 
"""The image shows a comparison of two skin conditions on what appears to be the back of a patient.  On the left is a brown, asymmetrical mole, which could indicate melanoma or dysplastic nevus. On the right is a round, hypopigmented patch that could be caused by tinea versicolor, vitiligo, or pityriasis alba. A dermatologist should evaluate these skin conditions for a proper diagnosis.""",
"""The image shows the lower body, specifically the legs and thighs, of an infant with dark skin. There are areas of erythema (redness) and scaling on the thighs. This could indicate a number of skin conditions, including:

* **Eczema (Atopic Dermatitis):** A common condition causing red, itchy, and inflamed skin. 
* **Contact Dermatitis:** An allergic reaction or irritant reaction causing a red, itchy rash.
* **Seborrheic Dermatitis:**  A common skin condition causing flaky, white to yellowish scales on oily areas, sometimes affecting infants.
* **Psoriasis:**  While less common in infants, it can cause red, scaly patches.
* **Fungal Infection:**  Certain fungal infections can cause red, itchy, and scaly rashes.""", """The image shows a close-up view of a person's cheek, which is characterized by numerous small, scattered, dark bumps and pits. These are indicative of a skin condition.

Possible diseases that could cause these symptoms include:

1. **Acne:** Acne often manifests as pimples, blackheads, and whiteheads, but it can also leave behind pitted scars, particularly if it was severe or not treated properly. 
2. **Folliculitis:** This condition involves inflammation of hair follicles, leading to small, red bumps that may be itchy or painful. 
3. **Keratosis pilaris:** Often referred to as "chicken skin," keratosis pilaris causes small, rough bumps due to a buildup of keratin in hair follicles. 
4. **Milia:** These are tiny white bumps that appear on the skin when dead skin cells become trapped beneath the surface.
5. **Post-inflammatory hyperpigmentation (PIH):**  PIH is a common consequence of skin inflammation, particularly in people with darker skin tones. It can cause flat, brown spots where acne or other skin lesions have healed.""", 
"""The image shows a hand with multiple large, fluid-filled blisters (bullae) on the fingers and palm. The blisters are tense and appear to be filled with clear fluid. These characteristics suggest:

* **Second-degree burn:** The presence of large blisters and location on the hand suggest a burn, likely caused by heat. Second-degree burns affect the top two layers of skin.
* **Friction burn:**  Similar in appearance to heat burns, friction burns result from rubbing or abrasion against a rough surface. 
* **Bullous impetigo:** While less likely, this bacterial infection can cause large, fluid-filled blisters. However, impetigo blisters often rupture easily. 
* **Allergic reaction:** Severe allergic reactions to certain substances can manifest as large blisters. """, 
"""The image shows the skin of a human face with multiple small black dots surrounded by red inflamed skin. Possible diseases include:

1. **Acne:** The presence of blackheads (open comedones) and inflamed papules point towards acne vulgaris.
2. **Folliculitis:** This condition involves inflammation of hair follicles and can also present with blackheads and red bumps. 
3. **Keratosis pilaris:** While less likely due to the presence of blackheads, this condition causes rough patches and small bumps on the skin. 
4. **Contact dermatitis:** Although less likely, an allergic reaction or irritant could potentially cause similar skin changes.""",
"""The image shows a close-up view of a skin condition characterized by a large, raised, erythematous (red) lesion on the leg. The lesion appears to be filled with fluid or pus, suggesting an infection or inflammatory process. 

Possible diagnoses based on the appearance include:

* **Cellulitis:** A bacterial skin infection that causes redness, swelling, warmth, and pain.
* **Abscess:** A localized collection of pus within the skin.
* **Insect bite reaction:**  Some insect bites can cause significant inflammation and swelling.
* **Infected cyst:** A cyst that has become infected can resemble the lesion shown.
""", 
"""The image shows a close-up view of a skin condition characterized by a large, raised, erythematous (red) lesion on the leg. The lesion appears to be filled with fluid or pus, suggesting an infection or inflammatory process. 

Possible diagnoses based on the appearance include:

* **Cellulitis:** A bacterial skin infection that causes redness, swelling, warmth, and pain.
* **Abscess:** A localized collection of pus within the skin.
* **Insect bite reaction:**  Some insect bites can cause significant inflammation and swelling.
* **Infected cyst:** A cyst that has become infected can resemble the lesion shown.""", 
"""The image shows a person's forehead with a large area of redness and several small, raised bumps. The skin appears inflamed. This could be indicative of several conditions, including:

* **Allergic reaction:** This is a common cause of skin redness and bumps, and can be triggered by a variety of substances, such as cosmetics, pollen, or food. 
* **Contact dermatitis:** This is a type of eczema that is triggered by contact with an irritant, such as soap, detergent, or poison ivy.
* **Atopic dermatitis (Eczema):** A chronic condition that causes dry, itchy, and inflamed skin. 
* **Rosacea:** A chronic skin condition that causes redness, flushing, and small, pus-filled bumps on the face.
* **Insect bites:** Bites from mosquitoes, bed bugs, or other insects can cause red, itchy bumps.
* **Folliculitis:** Inflammation of hair follicles, often caused by bacteria.""",
"""The image shows a close-up view of human skin, specifically the cheek, with a small, reddish papule with a central indentation. The surrounding skin appears normal.  Given its appearance, this lesion could be:

* **Basal cell carcinoma:**  While these often present as pearly or waxy bumps, they can sometimes have a reddish, irritated appearance. The central indentation is a concerning feature.
* **Squamous cell carcinoma:** These often appear as scaly, red patches, but they can also develop into firm nodules. 
* **Merkel cell carcinoma:** This rare and aggressive skin cancer often appears as a painless, firm, flesh-colored or bluish-red nodule.
* **Keratoacanthoma:** This benign (non-cancerous) growth often appears as a dome-shaped nodule with a central crater. 
* **Molluscum contagiosum:** This viral skin infection can cause flesh-colored to reddish papules with a central indentation.""",
"""The image shows a close-up view of the back of both knees. The knees exhibit areas of redness and scaling, with a larger, well-defined silvery-white plaque present on the right knee. This is characteristic of **psoriasis**, a chronic autoimmune skin condition. 

Other possible conditions to consider based on these features include:

* **Eczema (atopic dermatitis):**  While eczema can also cause redness and scaling, it typically presents with more intense itching and less well-defined borders.
* **Seborrheic dermatitis:** This condition typically affects areas rich in oil glands, such as the scalp and face, but it can occur on other body parts, including the knees. 
* **Lichen simplex chronicus:**  This condition is characterized by thickened, scaly plaques that develop due to repeated rubbing or scratching.""",
"""The image shows the back of a person's neck with widespread, fine, reddish-brown papules. The skin appears thickened. This could indicate:

* **Acanthosis nigricans:** This condition causes velvety, dark, thickened skin patches, often in skin folds like the back of the neck. It can be associated with insulin resistance, obesity, hormonal disorders, or certain medications.
* **Lichen planus:** This inflammatory skin condition can cause itchy, flat-topped, purplish bumps. While usually appearing on the wrists, ankles, and inside of the mouth, it can affect any area, including the neck.
* **Dermatitis neglecta:**  This condition is caused by poor hygiene and results in a buildup of dead skin cells, oil, and sweat, leading to a brown, scaly appearance.
* **Drug eruption:** Some medications can cause skin reactions that manifest as rashes or bumps.
* **Contact dermatitis:** This allergic reaction occurs upon contact with irritants or allergens, potentially causing redness, bumps, and itching.""",
"""The image shows a young child with significant erythema and edema affecting the face, particularly around the eyes. Possible diagnoses include: 

* **Allergic reaction:** This is a strong possibility, given the presentation and localization of the swelling. 
* **Angioedema:** This is a more severe form of allergic reaction that involves deeper layers of the skin.
* **Cellulitis:** This is a bacterial skin infection that can cause redness, swelling, and pain. 
* **Viral infection:** Several viruses, such as those causing measles or rubella, can cause a rash with facial swelling.""", 
"""The image shows a close-up view of a patient's face, specifically the area around the nose and mouth. The skin exhibits yellow crusting,  erosions, and surrounding erythema. This presentation could indicate several possible conditions, including:

* **Impetigo:** A common, highly contagious bacterial skin infection that often presents with honey-colored crusts.
* **Eczema herpeticum:** A more serious condition caused by the herpes simplex virus, commonly occurring in individuals with eczema. It can present with painful blisters and erosions.
* **Contact dermatitis:**  An inflammatory skin reaction that can be triggered by allergens or irritants, leading to redness, itching, and blisters.
* **Severe allergic reaction:** Certain severe allergic reactions can manifest with skin symptoms similar to those observed in the image.""",
"""The image shows a person's back with multiple hypopigmented macules and patches. Hypopigmentation means that the affected areas of skin are lighter than the surrounding skin. The lesions are of various sizes and shapes and are scattered across the entire back.  

This pattern of hypopigmentation could be indicative of several conditions, including:

* **Tinea versicolor (Pityriasis versicolor):** A fungal infection that often causes hypopigmented patches, particularly on the back, chest, and shoulders. 
* **Vitiligo:** An autoimmune disorder where the body attacks pigment-producing cells, leading to depigmented patches of skin.
* **Post-inflammatory hypopigmentation:**  This can occur after skin inflammation or injury, such as eczema, psoriasis, or burns. 
* **Lichen sclerosus:**  A long-term skin condition that can cause thin, white, itchy patches of skin.
* **Idiopathic guttate hypomelanosis:**  A harmless condition that causes small, white spots on the skin, often due to sun exposure.""",
"""The image shows a close-up view of a hand, likely a child's, afflicted with widespread erythema and a subtly  "velvety" texture. The affected area is dry and rough to the touch. These characteristics are consistent with several possible skin conditions, including:

* **Atopic dermatitis (eczema):** A chronic inflammatory skin condition characterized by dry, itchy, and inflamed skin.
* **Contact dermatitis:**  An inflammatory skin reaction triggered by contact with an allergen or irritant. This can present with redness, itching, and a rash that may be scaly or blistered.
* **Psoriasis:** A chronic autoimmune disease that causes the rapid buildup of skin cells, leading to thick, scaly, and itchy patches.
* **Scabies:** A contagious skin infestation caused by mites. It causes intense itching and a pimple-like rash. 
* **Drug reaction:** Certain medications can cause skin reactions as a side effect.""",
"""The image shows the back of a person's torso with a large, irregular-shaped area of redness and slight scaling.  

Possible diseases include:

* **Eczema (atopic dermatitis):** This is a common inflammatory skin condition that causes red, itchy, and dry skin. 
* **Contact dermatitis:**  This is a type of eczema triggered by direct contact with an irritant (like a harsh soap) or an allergen (like poison ivy).
* **Tinea versicolor:** This is a fungal infection that causes small, discolored patches of skin, often on the back and chest. 
* **Psoriasis:** This autoimmune disease causes raised, red, scaly patches of skin.""",
"""The image shows a close-up view of an ear with a  well-defined lesion that is  erythematous, crusted, and ulcerated. The surrounding skin is scaly and inflamed.  Possible diagnoses include: 
* **Basal cell carcinoma (BCC):**  Common skin cancer that often appears as a pearly or waxy bump, though it can also be a flat, flesh-colored or brown scar-like lesion. 
* **Squamous cell carcinoma (SCC)**: Type of skin cancer that can appear as a firm, red nodule, a scaly growth that bleeds, or a sore that doesn't heal.
* **Actinic keratosis (AK):** Precancerous skin lesion that can develop into SCC, usually appearing as a rough, scaly patch. 
* **Cutaneous horn:** A conical projection of keratin that can be associated with various underlying skin conditions.""",
"""The image shows a close-up view of a finger with severe nail and skin infections. The nail is thickened, discolored, and partially detached from the nail bed. The surrounding skin is red, inflamed, and shows signs of crusting and scaling. There is also a small lesion on the fingertip. 

This clinical presentation could indicate several conditions, including:

* **Onychomycosis (fungal nail infection):** This is the most likely cause given the nail's appearance.
* **Psoriasis:** Nail changes, including pitting, thickening, and discoloration, are common in psoriasis.
* **Eczema:** This can also cause inflammation and scaling around the nails.
* **Bacterial infection:** The redness and crusting could indicate a secondary bacterial infection. 
* **Lichen planus:** This inflammatory condition can affect the nails, causing thinning, ridging, and splitting.""",
"""The image shows a close-up of a person's right cheek, revealing multiple red, inflamed papules and pustules consistent with acne. 

Possible diseases include:

1. **Acne Vulgaris:** A common skin condition characterized by pimples, blackheads, and whiteheads. The presence of papules and pustules strongly suggests acne vulgaris.
2. **Rosacea:** Although less likely, rosacea can also cause redness and pimples, especially on the cheeks. 
3. **Perioral Dermatitis:** This condition presents with small, red bumps around the mouth and nose. While the image focuses on the cheek, the possibility of perioral dermatitis spreading to this area cannot be ruled out completely.""",
"""The image shows a close-up view of a dark brown, slightly raised lesion on a person's cheek. It appears to be well-defined and symmetrical. This could be indicative of several skin conditions, including:

* **Seborrheic keratosis:** A common, benign skin growth that often appears as a brown, black, or tan growth that may feel waxy or scaly.
* **Melanocytic nevus (mole):** A common type of skin growth that can vary in color, size, and shape. Most moles are harmless, but some can develop into melanoma. 
* **Dermatofibroma:**  A common, benign nodule that typically appears as a firm, reddish-brown growth on the skin.
* **Basal cell carcinoma:** A type of skin cancer that can present as a pearly or waxy bump. While less likely given the appearance, it is important to consider in the differential diagnosis.""",
"""The image shows a big toenail with a dark, brown-black, horizontal band across it. This band is called longitudinal melanonychia.  Possible causes for this finding include: 

* **Benign melanonychia:** This is the most common cause, particularly in individuals with darker skin tones. It happens when pigment-producing cells in the nail (melanocytes) become more active.
* **Nail matrix nevus:** This is a mole in the nail matrix, the area under the skin where the nail grows from. 
* **Subungual melanoma:** This is a type of skin cancer that develops in the nail matrix. While less common, it's crucial to rule out, especially if the band is new, changing, or affects only one nail.""",
"""The image shows a person's back with multiple, scattered, hypopigmented macules and patches. This appearance could be indicative of several skin conditions, including: 

* **Vitiligo:** An autoimmune disorder where the body attacks melanocytes, resulting in smooth, well-defined white patches.
* **Pityriasis alba:** A common skin condition characterized by poorly defined, hypopigmented, and slightly scaly patches. It often occurs in children and young adults.
* **Tinea versicolor:** A fungal infection that causes hypopigmented or hyperpigmented patches, often with fine scaling. 
* **Post-inflammatory hypopigmentation:** Lightening of the skin following inflammation or injury, such as eczema or burns. """,
"""The image shows an area of skin on the arm with redness, scaling, and excoriation marks, which are indicative of scratching. Several conditions could cause these symptoms, including:

* **Eczema (Atopic Dermatitis):** A common skin condition characterized by dry, itchy, and inflamed skin. 
* **Allergic Contact Dermatitis:** An itchy rash that develops when the skin comes into contact with an allergen.
* **Psoriasis:** An autoimmune disease that causes the rapid buildup of skin cells, resulting in thick, scaly patches.
* **Scabies:** A contagious skin infestation caused by mites that burrow into the skin.
* **Fungal Infection:** An infection of the skin caused by a fungus, which can lead to redness, itching, and scaling. 
""", """The image shows a close-up view of a person's face, specifically focusing on the area around the eyes. The most prominent feature is the presence of depigmented, or lighter, patches of skin on both eyelids. These patches are irregular in shape and appear smooth and non-scaly.

Given the location and appearance of the skin changes, the following possible conditions could be considered:

1. **Vitiligo:** This is the most likely diagnosis, characterized by the loss of melanin (skin pigment) in patches. It can affect any part of the body, including the eyelids.

2. **Pityriasis alba:** This condition typically presents as hypopigmented (lighter) patches, often on the face, especially in children and young adults. However, it usually involves slightly scaly patches, which are not evident in this image.

3. **Post-inflammatory hypopigmentation:** This can occur after skin inflammation or injury, leading to temporary or permanent loss of pigmentation.  

4. **Nevus depigmentosus:** This is a rare birthmark characterized by a well-defined patch of hypopigmentation. It is usually present at birth or appears in early childhood.

5. **Leprosy (in some cases):** Certain types of leprosy can cause hypopigmented patches with sensory loss.""", """The image shows a close-up view of the skin on a person's face, specifically the area around the nose and under the eye. There are two small, raised, flesh-colored lesions. The lesion closer to the nose is larger and has a central indentation. 

Possible diagnoses based on these lesions include:

* **Sebaceous hyperplasia:** These are benign (non-cancerous) growths of the sebaceous glands, which produce oil. They are common in adults and often appear as small, yellowish bumps with a central indentation.
* **Basal cell carcinoma:**  While less likely given the appearance, it's important to consider skin cancer. Basal cell carcinomas can present as pearly or waxy bumps, sometimes with a central indentation. 
* **Molluscum contagiosum:** This is a viral skin infection that causes small, flesh-colored bumps with a central indentation. However, these bumps are usually pearly and often have a white plug in the center.
* **Flat warts:** These are small, flat-topped warts that can appear anywhere on the body. They are caused by a virus and are contagious.""",
"""The image shows a close-up view of the skin on the neck of a person. There is a single, small, pink, dome-shaped bump (papule) present. The surrounding skin is wrinkled and shows signs of sun damage.

Possible diseases associated with this image include:

1. **Basal Cell Carcinoma (BCC):** BCCs are common skin cancers that often appear as pearly or waxy bumps. While the bump in the image is not classic for BCC, it's important to consider in the differential diagnosis.
2. **Squamous Cell Carcinoma (SCC):** SCCs are another type of skin cancer that can present as scaly red patches, open sores, or elevated growths. While the bump in the image is not typical for SCC, it's worth considering, especially given the sun-damaged skin.
3. **Merkel Cell Carcinoma (MCC):** MCC is a rare but aggressive type of skin cancer. It typically presents as a rapidly growing, firm, shiny nodule that may be flesh-colored or bluish-red. This should be considered, although the bump in the image doesn't perfectly match the typical appearance of MCC.
4. **Intradermal Nevus (Mole):** An intradermal nevus is a benign growth of melanocytes (pigment-producing cells) located in the deeper layers of the skin. These moles can sometimes appear as dome-shaped papules.
5. **Dermal Cyst:** A dermal cyst is a noncancerous, closed sac under the skin that contains fluid or semisolid material. They can sometimes appear as smooth, dome-shaped bumps.
6. **Other Benign Growths:** Other possibilities include a dermatofibroma (a benign fibrous growth) or a neurofibroma (a benign tumor of nerve tissue). 
""",
"""The image shows a close-up view of a hand with dark, hyperpigmented skin, particularly on the back of the hand and fingers. The skin appears thick and leathery with accentuated skin lines. 

Given the provided information, the appearance of the hand could be attributed to various conditions.  Here are some possibilities:

* **Acanthosis nigricans:** This condition causes velvety, darkened skin in body folds and creases. It's often associated with insulin resistance, obesity, and hormonal disorders. 
* **Chronic eczema:** Long-term eczema can lead to skin thickening, discoloration (hyperpigmentation), and prominent skin markings.
* **Scleroderma:** While less likely, scleroderma can cause skin thickening and hardening. It can also affect other organs, including blood vessels and internal organs.
* **Drug-induced hyperpigmentation:** Certain medications can cause skin darkening as a side effect.
* **Post-inflammatory hyperpigmentation:** This is a common response to skin injury or inflammation where the skin produces excess melanin. """,
"""The image shows the upper back of a person with several brown moles of varying sizes and shapes. Some moles are darker than others. 

Possible conditions associated with moles include:

* **Normal moles (benign nevi):** Most moles are harmless and don't require treatment. 
* **Atypical moles (dysplastic nevi):** These moles may have irregular features like uneven color, blurry borders, or larger size. They have a higher risk of becoming melanoma.
* **Melanoma:** This is the most serious type of skin cancer. It can develop from existing moles or appear as new spots.  """, 
"""The image shows an arm with a rash in the elbow crease. The rash is red, dry, and scaly. It could be a sign of several skin conditions like eczema, psoriasis, contact dermatitis, or fungal infection.""",
"""The image shows multiple red and purple, fluid-filled blisters on a person's upper chest/shoulder area. Some blisters appear to be scabbed. This could be a symptom of several things such as: 

1. **Shingles (Herpes Zoster)**: Shingles often presents as a painful rash with blisters on one side of the body. 
2. **Bullous impetigo**: A bacterial skin infection causing large, fluid-filled blisters. 
3. **Allergic reaction**:  Certain allergies can manifest as blisters and skin irritation. 
4. **Insect bites**:  Multiple bites can result in clustered, itchy blisters. """,
"""This is an image of the left ear and the surrounding skin behind the ear of a fair-skinned individual. There is no obvious rash, discoloration, or other abnormalities of the skin. 
""", """The image shows an arm with multiple excoriated papules and erosions. Possible diseases include: 
* **Atopic dermatitis (eczema):** A chronic inflammatory skin condition that causes itchy, red, and dry skin.  
* **Contact dermatitis:** An itchy rash caused by direct contact with an irritant or an allergic reaction to a substance. 
* **Scabies:** A highly contagious skin infestation caused by a mite that burrows under the skin, leading to intense itching and a pimple-like rash. 
* **Insect bites:**  Bites from insects like mosquitoes, bed bugs, or fleas can cause itchy, red bumps or welts.""",
"""The image shows a close-up view of the skin around a person's eye. There are numerous small, white or yellowish, firm bumps on the skin.  This appearance could indicate several conditions, including: 

* **Milia:** These are very common, harmless cysts filled with keratin. They are especially common in newborns but can occur at any age. 
* **Syringomas:** These are benign sweat duct tumors that typically appear as small, flesh-colored or yellowish bumps. They often occur in clusters around the eyes.
* **Sebaceous hyperplasia:** This condition involves enlarged oil glands and can result in yellowish, dome-shaped bumps on the skin, often on the face.
* **Flat warts:**  These are small, flat-topped warts that can appear anywhere on the body but are common on the face.""",
"""The image shows a close-up view of a person's face, specifically around the mouth area. There is a noticeable depigmented patch of skin present. 

Based on the appearance, the most likely condition is **vitiligo**, which causes loss of skin color in patches. 

However, other possibilities to consider include:

* **Pityriasis alba:** This condition often affects children and presents as light, scaly patches on the skin.
* **Tinea versicolor:** This fungal infection can cause lighter or darker patches on the skin.
* **Post-inflammatory hypopigmentation:** This can occur after skin injury or inflammation, leaving behind a lighter area of skin.""",
"""The image shows a close-up view of the back of a person's hand with dark skin. The hand has numerous small, scattered, white papules.  This could indicate a few possible conditions:

* **Lichen planus:**  Characterized by itchy, flat-topped, purplish papules. The appearance in this image could be a variant of lichen planus.
* **Molluscum contagiosum:**  Caused by a poxvirus, this condition presents with small, flesh-colored or pearly-white, dome-shaped papules, often with a central indentation.
* **Flat warts:**  These are small, smooth, flat-topped warts that are typically flesh-colored or slightly darker than the surrounding skin.
* **Folliculitis:** This is an inflammation of the hair follicles, which can cause small, white-headed pimples.""",
"""The image shows a close-up view of a skin lesion. The lesion is a raised, flesh-colored papule with a smooth, pearly surface. There is some blood present on the lesion, indicating recent trauma or irritation. This could be due to a number of things, including scratching or picking at the lesion. 

Given the appearance and characteristics, here are a few possibilities of what this skin lesion could be:

* **Basal cell carcinoma:**  While basal cell carcinomas can sometimes bleed, they usually present with a pearly or waxy appearance, often with visible blood vessels. 
* **Squamous cell carcinoma:** This type of skin cancer can also bleed and might appear as a firm, red nodule or a rough, scaly lesion. 
* **Molluscum contagiosum:** This viral infection often presents as small, flesh-colored or pearly-white bumps with a central dimple.  
* **Skin tag (acrochordon):** Skin tags are harmless, flesh-colored growths that often occur in areas of friction, such as the neck, armpits, or groin. 
* **Insect bite:** Some insect bites can cause red, swollen bumps that may bleed.""",
"""The image shows the back of a hand with multiple, well-defined, hypopigmented macules and patches. Some lesions are confluent and show fine scaling. The depigmentation is not complete, with some areas retaining a light brown pigmentation. 

Possible diagnoses include:

* **Vitiligo:** An autoimmune disorder causing loss of skin pigmentation. 
* **Pityriasis alba:** A common, benign skin condition characterized by hypopigmented patches, often occurring in children and adolescents. 
* **Tinea versicolor (pityriasis versicolor):** A fungal infection that can cause hypopigmented or hyperpigmented patches, often on the trunk and upper arms but can occur on the hands.
* **Post-inflammatory hypopigmentation:**  Lightening of the skin after inflammation or injury, such as eczema or burns.""",
"""The image shows a patch of hair loss on the scalp. This is a nonspecific finding and could be caused by a variety of conditions, including:

* **Alopecia areata:** An autoimmune disorder that causes patchy hair loss. This is the most likely diagnosis in this case, given the well-defined patch of hair loss.
* **Tinea capitis (ringworm of the scalp):** A fungal infection that can cause hair loss, scaling, and inflammation. 
* **Trichotillomania:** A hair-pulling disorder that can lead to patchy hair loss.
* **Scarring alopecia:** A group of conditions that cause hair loss and scarring of the scalp.
* **Telogen effluvium:** A type of hair loss that occurs after a stressful event.""", """This image shows the inside of an ear canal with a dark, scaly lesion present. 

Possible diseases include:

* **Otitis Externa (Swimmer's Ear):** This is an inflammation of the ear canal, often caused by infection. While this image doesn't show typical redness and swelling, some forms of otitis externa can present with scaling and debris.
* **Eczema:** This inflammatory skin condition can affect the ear canal, causing itching, redness, scaling, and cracking.
* **Psoriasis:** Another skin condition that can cause thick, scaly patches, although the appearance here isn't classic for psoriasis.
* **Fungal Infection (Otomycosis):**  The dark color and scaly appearance could indicate a fungal infection of the ear canal.
* **Seborrheic Dermatitis:** This common skin condition can cause flaky, white to yellowish scales, often on the scalp but it can affect areas like the ear canal. 
* **Skin Cancer:** While less likely, any unusual lesion warrants investigation to rule out skin cancer. """,
"""The image shows a single, well-defined, erythematous plaque on the leg.  The lesion is slightly raised and scaly.  Possible diagnoses include:

* **Nummular eczema:** This is a common type of eczema that causes coin-shaped, itchy, and scaly patches of skin. 
* **Tinea corporis (ringworm):** This is a fungal infection of the skin that can cause red, ring-shaped rashes.
* **Psoriasis:**  This is a chronic autoimmune condition that can cause thick, scaly, red patches of skin.
* **Drug eruption:** Some medications can cause a variety of skin reactions, including red, itchy rashes."""]


import pinecone
import os

# %%
load_dotenv(r'C:\Users\91982\Desktop\Taskformer\.env')
p= os.getenv('P')

# %%
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
pc = Pinecone(api_key=p)

# %%
index_name= "nemo"
if index_name not in pc.list_indexes().names():
    
    pc.create_index(
        name=index_name,
        dimension=384, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )


# %%
index= pc.Index(index_name)

# %%
index.describe_index_stats()

# %% [markdown]
# ### Make embeddings to store

# %%
import torch

from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')


full_data=dataset1+dataset2+actual_desc+ai_desc

print(len(full_data))
def write_full_data_to_file(full_data, file_path):
    with open(file_path, 'w',encoding='utf-8') as file:
        for item in full_data:
            file.write(f"{item}\n")


file_path = "full_data.txt"
write_full_data_to_file(full_data, file_path)



# %%
# from tqdm.auto import tqdm

# def embed_and_upsert_texts(texts, batch_size=50):
#     for i in tqdm(range(0, len(texts), batch_size)):
#         batch_texts = texts[i:i + batch_size]
#         embeddings = model.encode(batch_texts)
#         ids = [f'id-{i + j}' for j in range(len(batch_texts))]
#         vectors = list(zip(ids, embeddings))
#         index.upsert(vectors)

# # Embed and upsert texts in batches
# embed_and_upsert_texts(full_data, batch_size=50)

# print("All embeddings stored in Pinecone.")

# %%
from langchain_community.chat_models import JinaChat
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)



load_dotenv(r'C:\Users\91982\Desktop\Taskformer\.env')


j = os.getenv('J')



def generate_response(messages, jinachat_api_key):
    chat = JinaChat(temperature=0, jinachat_api_key=j)
    
    # Generate a response using JinaChat
    response = chat(messages)
    
    return response.content


def query_documents(query_text, top_k=5):

    query_embedding = model.encode([query_text])[0]
    query_results = index.query(query_embedding.tolist(), top_k=top_k)
    

    matching_ids = [match['id'] for match in query_results['matches']]
    return matching_ids


def create_id_to_text_mapping(full_data):
    return {f'id-{i}': text for i, text in enumerate(full_data)}


id_to_text = create_id_to_text_mapping(full_data)


