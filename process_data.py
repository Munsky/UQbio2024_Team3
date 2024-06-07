def process_data(filepath,figName):

    # Loading data
    img = imread(str(figName))

    img_cyto_max = np.max(img[:, :, :, 1], axis = 0)
    model1 = models.Cellpose(gpu = False, model_type = 'cyto')

    # Apply the model to your image
    masks_cyto, flows, styles, diams = model1.eval(img_cyto_max, diameter=None, channels=[0,0])
    # Plotting each one of the 3 colors independently
    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    ax[0].imshow(masks_cyto,cmap='Spectral')
    ax[0].axis('off')
    ax[0].set_title('Cell Segmentation')

    img_nuc = img[0, :, :, 0]
    model2 = models.Cellpose(gpu = False, model_type = 'cyto')

    # Apply the model to your image
    masks_nuc, flows, styles, diams = model2.eval(img_nuc, diameter=None, channels=[0,0])
    # Plotting each one of the 3 colors independently
    ax[1].imshow(masks_nuc,cmap='Spectral')
    ax[1].axis('off')
    ax[1].set_title('Nucleus Segmentation')
    # Save the figure
    plt.savefig(os.path.join(filepath, 'seg_result.png'), dpi=300)

    # %%
    # Make sure that one cell has the same id in both masks
    # Get the unique labels in the cell and nucleus masks
    cell_labels = np.unique(masks_cyto)
    nucleus_labels = np.unique(masks_nuc)

    # Initialize the mapping table
    mapping_table = {}

    # Loop through each cell label
    for cell_label in cell_labels:
        # Skip the background
        if cell_label == 0:
            continue

        # Get the mask for the current cell
        current_cell_mask = (masks_cyto == cell_label)

        # Loop through each nucleus label
        for nucleus_label in nucleus_labels:
            # Skip the background
            if nucleus_label == 0:
                continue

            # Get the mask for the current nucleus
            current_nucleus_mask = (masks_nuc == nucleus_label)

            # If the current cell and the current nucleus overlap, add them to the mapping table
            if np.any(current_cell_mask & current_nucleus_mask):
                mapping_table[cell_label] = nucleus_label
                break
    updated_masks_nuc = masks_nuc.copy()

    # Loop through each item in the mapping table
    for cell_label, nucleus_label in mapping_table.items():
        # Update the nucleus label in the nucleus mask
        updated_masks_nuc[masks_nuc == nucleus_label] = cell_label
    masks_nuc = updated_masks_nuc

    # %% [markdown]
    # calculate the number of mRNAs in each nuclei and cytoplasm

    # %%
    # mRNA identification
    threshold_m = 0.025
    cell_id = np.unique(masks_cyto)
    t_total = img.shape[0]

    df_m_c = pd.DataFrame()
    df_m_n = pd.DataFrame()

    df_m_c['cell id'] = cell_id[1:]
    df_m_n['cell id'] = cell_id[1:]

    for t in range(t_total):
        # get the area of a single mRNA
        mRNA = img[t, :, :, 2]
        mRNA_filtered = difference_of_gaussians(mRNA, low_sigma=1, high_sigma=5)

        mRNA_binary = mRNA_filtered.copy()
        mRNA_binary[mRNA_binary>=threshold_m] = threshold_m # Making spots above the threshold equal to the threshold value.
        mRNA_binary[mRNA_binary<threshold_m] = 0 # Making spots below the threshold equal to 0.

        mRNA_binary[mRNA_binary!=0] = 1 # Binarization

        spot_contours = measure.find_contours(mRNA_binary, 0.9)

        labels_m = measure.label(mRNA_binary)

        props = measure.regionprops(labels_m, intensity_image=mRNA)

        # Initialize a list to store the intensities
        areas = []

        # Loop through each mRNA in the image
        for prop in props:
            # Calculate the intensity of the current mRNA and add it to the list

            areas.append(prop.area)

        areas = np.array(areas)
        areas_sorted = np.sort(areas)
        start = np.percentile(areas_sorted, 10)
        end = np.percentile(areas_sorted, 50)
        selected_areas = areas_sorted[(areas_sorted>=start) & (areas_sorted<=end)]
        single_mRNA_area = np.mean(selected_areas)

        mRNA_num_cyto = []
        mRNA_num_nuc = []

        for id in cell_id:
            # Skip the background
            if id == 0:
                continue

            # Get the mask for the current cell
            nuc_mask = (masks_nuc == id)

            # Calculate the properties of the mRNAs in the current cell
            props_n = measure.regionprops(labels_m * nuc_mask, intensity_image = mRNA)

            # Initialize the total area, total intensity and count for the current cell
            total_area_n = 0
            count_n = 0

            # Loop through each mRNA in the current cell
            for prop in props_n:
                # Update the total area, total intensity and count
                total_area_n += prop.area

            count_n = round(total_area_n / single_mRNA_area)
            # Add the results for the current cell to the DataFrame

            mRNA_num_nuc.append(count_n)
            
            # Get the mask for the current cell
            mask_c = (masks_cyto == id)
            mask_n = (masks_nuc == id)
            cyto_mask = mask_c & ~mask_n

            # Calculate the properties of the mRNAs in the current cell
            props_c = measure.regionprops(labels_m * cyto_mask, intensity_image = mRNA)

            # Initialize the total area, total intensity and count for the current cell
            total_area_c = 0
            count_c = 0

            # Loop through each mRNA in the current cell
            for prop in props_c:
                # Update the total intensity and count
                total_area_c += prop.area

            count_c = round(total_area_c / single_mRNA_area)
            # Add the results for the current cell to the DataFrame

            mRNA_num_cyto.append(count_c)

        df_m_c['frame' + str(t+1)] = mRNA_num_cyto
        df_m_n['frame' + str(t+1)] = mRNA_num_nuc


    # %%
    # Save the DataFrame to a csv file
    df_m_n.to_csv(os.path.join(filepath, 'mRNAs_in_nucleus.csv'), index=False)
    df_m_c.to_csv(os.path.join(filepath,'mRNAs_in_cyto.csv'), index=False)

    # %% [markdown]
    # number and intensity of transcription site in each cell

    # %%
    trans_site = np.max(img[:, :, :, 2], axis=0)

    # Apply a threshold to segment the transition sites
    threshold = filters.threshold_otsu(trans_site)
    binary_trans = trans_site > threshold

    # Label connected components
    labeled_trans, num_labels = ndi.label(binary_trans)

    plt.figure(figsize=(5,5))
    plt.imshow(labeled_trans, cmap='nipy_spectral')
    plt.colorbar()
    plt.title('trans_sites')
    # Save the figure
    plt.savefig(os.path.join(filepath,'trans_sites.png'), dpi=300)

    # %%
    # Initialize an empty DataFrame
    df_trans = pd.DataFrame()

    # Initialize an empty list to store the cell each region belongs to
    cell_labels = []

    # Loop over each frame in the image
    for i in range(img.shape[0]):
        # Calculate the properties of each region for the current frame
        regions = measure.regionprops(labeled_trans, intensity_image=img[i,:,:,2])
        
        # Get the mean intensity of each region
        brightness = [(region.mean_intensity)*(region.area) for region in regions]
        
        # Add the brightness values to the DataFrame
        df_trans['frame'+str(i+1)] = brightness

    # Loop over each region
    for region in regions:
        # Get the coordinates of the region
        coords = region.coords
        
        # Get the labels of the cells that the region belongs to in masks_cyto
        labels = masks_cyto[coords[:,0], coords[:,1]]
        
        # Get the most common label, which is the cell that the region belongs to
        cell_label = np.argmax(np.bincount(labels))
        
        # Add the cell label to the list
        cell_labels.append(cell_label)

    # Add the cell labels to the DataFrame
    df_trans.insert(0, 'cell', cell_labels)

    # %%
    # Count the number of transcription sites for each cell
    transcription_sites_per_cell = df_trans['cell'].value_counts()
    for i in cell_id[1:]:
        if i not in transcription_sites_per_cell.index:
            transcription_sites_per_cell[i] = 0
    sorted_trans = transcription_sites_per_cell.sort_index()
    num_trans = sorted_trans.tolist()

    # %%
    # Save the DataFrame to a csv file
    df_trans.to_csv(os.path.join(filepath,'trans.csv'), index=False)

    # %% [markdown]
    # Cell size, nucleus size and number of transcription sites

    # %%
    cell_size = []
    nucleus_size = []
    trans_num = []

    cell_id = np.unique(masks_cyto)
    nucleus_id = np.unique(masks_nuc)

    for id in cell_id:
        if id == 0:
            continue
        cell_size.append(np.sum(masks_cyto == id))

    for id in nucleus_id:
        if id == 0:
            continue
        nucleus_size.append(np.sum(masks_nuc == id))

    df_info = pd.DataFrame({
        'cell_id': cell_id[1:],
        'cell_size': cell_size,
        'nucleus_size': nucleus_size,
        'trans_num': num_trans,
    })

    df_info.to_csv(os.path.join(filepath,'Basic_Information.csv'), index=False)

    # %% [markdown]
    # Protein Intensity

    # %%
    #Protein Intensity
    #createDataFrame
    df_protein = pd.DataFrame()

    # t_total = img.shape[0]

    for t in range(t_total):
        # If the frame has multiple channels, use only the first channel
        frame = img[t, :, :, 1]
        df_protein['cell id'] = np.nan
        df_protein['frame'+str(t+1)] = np.nan
        # Get unique cell masks
        unique_masks = np.unique(masks_cyto)
        
        # Loop through each unique mask (cell)
        for cell_id in unique_masks:
            if cell_id == 0:
                continue  # Skip background
            
            # Create a mask for the current cell
            cell_mask = (masks_cyto == cell_id).astype(np.uint8)
            
            # Check the sizes of the frame and cell_mask
            if frame.shape != cell_mask.shape:
                print(f"Size mismatch: frame shape {frame.shape}, cell_mask shape {cell_mask.shape}")
                continue
            
            # Calculate the mean intensity of the current cell
            total_intensity = np.sum(frame * cell_mask)
            area = np.sum(cell_mask)
            mean_intensity = total_intensity / area if area > 0 else 0
            
            # Append the results to the list
            df_protein.loc[cell_id-1, 'cell id'] = int(cell_id)
            df_protein.loc[cell_id-1, 'frame'+str(t+1)] = mean_intensity

    # Save the DataFrame
    df_protein.to_csv(os.path.join(filepath,'protein_intensity.csv'), index=False)