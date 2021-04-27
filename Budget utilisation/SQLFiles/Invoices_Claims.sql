SELECT invoice.member_id    
     , invoice.invoice_number
     , invoice.invoice_date
     , invoice.invoice_total
     , invoice.state invoice_state
     , invoice.is_reimbursement
     , claim.id claim_id
     , claim.claim_key     
     , claim.start_date claim_start_data
     , claim.end_date claim_end_date
     , claim.unit_price
     , claim.state claim_state
     , claim.invoiced_units
     , claim.unit_price
     , claim.claimed_date
     , claim.item_category_level3_id
     , claim.payment_id
     , claim.received_date
     , claim.submitted_date
     , claim.processed_status
     , claim.payment_id
     , level3.item_category_level2_id
     , level2.name
     , level2.code level2_code
     , level2.item_category_level1_id
     , level1.id
     , level1.name
     , level1.code level1_code
     , invoice.provider_account_id
  FROM HH_invoice invoice
                   
   LEFT JOIN HH_claim claim
			JOIN HH_item_category_level3 level3
			  ON level3.id = claim.item_category_level3_id
			JOIN HH_item_category_level2 level2
			  ON level2.id = level3.item_category_level2_id
			JOIN HH_item_category_level1 level1
			  ON level1.id = level2.item_category_level1_id
	      ON claim.invoice_id = invoice.id
         AND invoice.is_reimbursement = 0
         
 WHERE YEAR(invoice.created_at) = 2021
   AND YEAR(claim.created_at) = 2021
   ORDER BY invoice.member_id
          , invoice.invoice_date


